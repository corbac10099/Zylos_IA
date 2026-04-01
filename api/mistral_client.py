"""
api/mistral_client.py — Client Mistral API centralisé de ZYLOS AI
===================================================================
Usage EXCLUSIF : modules/improver.py (analyse et modification du code source).
Ce client ne doit JAMAIS être appelé pour répondre à l'utilisateur
ni pour générer des embeddings — ces tâches appartiennent à RWKV local.

Fonctionnalités :
  - Rate limiting strict (free tier ~1 req/sec) avec retry exponentiel
  - Compteur de tokens mensuels avec alerte à 80% du quota
  - Mode dégradé automatique si l'API est indisponible
  - Cache mémoire des appels identiques (évite les doublons)
  - Métriques complètes exportées vers utils/metrics.py

Usage :
    from api.mistral_client import mistral
    result = mistral.complete_code(code_source, metrics_json, context)
    if result.success:
        print(result.content)
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class MistralResponse:
    """
    Résultat d'un appel à l'API Mistral.

    Attributes:
        success:       True si l'appel a abouti sans erreur.
        content:       Texte retourné par le modèle (vide si échec).
        model:         Identifiant du modèle ayant répondu.
        tokens_in:     Tokens d'entrée consommés.
        tokens_out:    Tokens de sortie générés.
        latency_ms:    Durée de l'appel en millisecondes.
        error:         Message d'erreur si success=False.
        cached:        True si la réponse vient du cache local.
    """
    success:    bool
    content:    str   = ""
    model:      str   = ""
    tokens_in:  int   = 0
    tokens_out: int   = 0
    latency_ms: float = 0.0
    error:      str   = ""
    cached:     bool  = False


@dataclass
class _MonthlyUsage:
    """Compteur de tokens mensuel persisté dans les métriques."""
    tokens: int = 0
    month:  str = ""   # format "YYYY-MM"

    def reset_if_new_month(self) -> None:
        from datetime import datetime
        current = datetime.utcnow().strftime("%Y-%m")
        if self.month != current:
            self.tokens = 0
            self.month  = current


# ══════════════════════════════════════════════════════════════════════
# CLIENT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class MistralClient:
    """
    Client Mistral API thread-safe avec rate limiting et retry.

    Ce client est un singleton exposé via `mistral` en bas de fichier.
    Il gère :
      - La cadence des requêtes (max 1/sec sur le free tier)
      - Le retry exponentiel sur erreur transitoire (429, 5xx)
      - L'alerte quota mensuel
      - Le cache en mémoire pour éviter les appels redondants
      - Le mode dégradé si la clé est absente ou l'API injoignable

    Attributes:
        _lock:          Verrou pour l'accès concurrent aux métriques.
        _rate_lock:     Verrou dédié au rate limiting.
        _last_call_ts:  Timestamp UNIX du dernier appel réel à l'API.
        _usage:         Compteur de tokens mensuels.
        _cache:         Dictionnaire {hash_prompt: MistralResponse}.
        _degraded:      True si le mode dégradé est actif.
        _call_stats:    Compteurs d'appels par modèle.
        _latencies:     Liste des dernières latences pour la moyenne glissante.
    """

    _MAX_LATENCY_HISTORY = 50

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._rate_lock    = threading.Lock()
        self._last_call_ts = 0.0
        self._usage        = _MonthlyUsage()
        self._cache: dict[str, MistralResponse] = {}
        self._degraded     = False
        self._call_stats: dict[str, int] = {}
        self._latencies: list[float] = []

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def complete_code(
        self,
        module_code:  str,
        metrics_json: str,
        context:      str = "",
        use_cache:    bool = True,
    ) -> MistralResponse:
        """
        Envoie une demande d'analyse/amélioration de code à Mistral Codestral.

        C'est le SEUL point d'entrée autorisé depuis modules/improver.py.
        Le prompt est formaté selon la spec du Master Prompt ZYLOS.

        Args:
            module_code:  Code source du module à analyser (tronqué si > max_code_chars).
            metrics_json: Métriques actuelles du module au format JSON string.
            context:      Contexte additionnel (historique des modifications, etc.).
            use_cache:    Si True, retourne la réponse en cache si disponible.

        Returns:
            MistralResponse avec le JSON de suggestions ou une erreur.

        Example:
            >>> resp = mistral.complete_code(code, metrics_str)
            >>> if resp.success:
            ...     suggestions = json.loads(resp.content)
        """
        from config import MISTRAL

        if not MISTRAL.is_configured():
            log.warning(
                "MISTRAL_API_KEY non configurée — improver désactivé. "
                "Définissez la variable d'environnement MISTRAL_API_KEY."
            )
            return MistralResponse(
                success=False,
                error="MISTRAL_API_KEY absente — mode dégradé.",
                model=MISTRAL.code_model,
            )

        code_to_send = module_code
        if len(module_code) > MISTRAL.max_code_chars:
            code_to_send = module_code[:MISTRAL.max_code_chars]
            log.debug(
                "Code tronqué de %d à %d caractères pour l'API.",
                len(module_code), MISTRAL.max_code_chars,
            )

        prompt = self._build_improvement_prompt(code_to_send, metrics_json, context)

        if use_cache:
            cached = self._get_cached(prompt)
            if cached is not None:
                log.debug("Réponse Codestral retournée depuis le cache.")
                return cached

        return self._call_with_retry(
            model       = MISTRAL.code_model,
            prompt      = prompt,
            system      = self._CODESTRAL_SYSTEM,
            max_tokens  = 1000,
            temperature = 0.1,
            use_cache   = use_cache,
        )

    def is_available(self) -> bool:
        """
        Retourne True si l'API est configurée et non en mode dégradé.

        Returns:
            True si le client peut effectuer des appels.
        """
        from config import MISTRAL
        return MISTRAL.is_configured() and not self._degraded

    def get_metrics(self) -> dict[str, Any]:
        """
        Retourne un snapshot des métriques du client.

        Returns:
            Dictionnaire avec tokens, latences, compteurs d'appels, etc.
        """
        with self._lock:
            self._usage.reset_if_new_month()
            avg_lat = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies else 0.0
            )
            from config import MISTRAL
            budget_used_pct = (
                self._usage.tokens / MISTRAL.monthly_token_budget
                if MISTRAL.monthly_token_budget else 0.0
            )
            return {
                "tokens_this_month":   self._usage.tokens,
                "budget_used_pct":     round(budget_used_pct, 4),
                "budget_alert_active": budget_used_pct >= MISTRAL.alert_threshold_pct,
                "avg_latency_ms":      round(avg_lat, 1),
                "calls_by_model":      dict(self._call_stats),
                "cache_entries":       len(self._cache),
                "degraded_mode":       self._degraded,
                "month":               self._usage.month,
            }

    def reset_degraded(self) -> None:
        """
        Réinitialise le mode dégradé (à appeler après une panne API résolue).
        """
        with self._lock:
            self._degraded = False
        log.info("Mode dégradé réinitialisé — prochains appels tentés normalement.")

    def clear_cache(self) -> int:
        """
        Vide le cache mémoire des réponses.

        Returns:
            Nombre d'entrées supprimées.
        """
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
        log.debug("Cache Mistral vidé (%d entrées supprimées).", n)
        return n

    # ──────────────────────────────────────────────────────────────────
    # PROMPT SYSTÈME CODESTRAL
    # ──────────────────────────────────────────────────────────────────
    _CODESTRAL_SYSTEM = (
        "Tu es un expert Python senior spécialisé dans les systèmes IA non-Transformer. "
        "Tu analyses du code de production et proposes des améliorations ciblées. "
        "Tu réponds UNIQUEMENT en JSON valide, sans markdown, sans preamble, sans commentaire. "
        "Tes modifications respectent TOUJOURS les signatures publiques existantes."
    )

    @staticmethod
    def _build_improvement_prompt(
        code: str,
        metrics_json: str,
        context: str,
    ) -> str:
        """
        Construit le prompt d'amélioration selon la spec du Master Prompt.

        Args:
            code:         Code source tronqué du module.
            metrics_json: Métriques actuelles au format JSON string.
            context:      Historique des modifications et contraintes spécifiques.

        Returns:
            Chaîne de prompt complète à envoyer à Codestral.
        """
        parts = [
            "CONTEXTE : Module de ZYLOS AI (IA non-Transformer auto-apprenante, Python 3.10+).",
            "",
            f"MÉTRIQUES ACTUELLES :\n{metrics_json}",
        ]
        if context:
            parts += ["", f"HISTORIQUE / CONTRAINTES :\n{context}"]
        parts += [
            "",
            f"CODE SOURCE :\n{code}",
            "",
            "CONTRAINTES ABSOLUES :",
            "- Répondre UNIQUEMENT en JSON valide (pas de markdown, pas de backticks)",
            "- Ne modifier QUE les corps de méthodes existantes",
            "- Le code Python doit être syntaxiquement parfait",
            "- Respecter les types existants et les signatures publiques",
            "- Si aucune amélioration pertinente : retourner changes=[]",
            "",
            "FORMAT RÉPONSE (JSON strict) :",
            '{"analysis": "...", '
            '"changes": [{"type": "replace_function", "function_name": "...", '
            '"reason": "...", "new_code": "..."}], '
            '"priority": "high|medium|low", '
            '"estimated_improvement": "..."}',
        ]
        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────────────
    # APPEL HTTP AVEC RETRY EXPONENTIEL
    # ──────────────────────────────────────────────────────────────────
    def _call_with_retry(
        self,
        model:       str,
        prompt:      str,
        system:      str,
        max_tokens:  int,
        temperature: float,
        use_cache:   bool,
    ) -> MistralResponse:
        """
        Exécute l'appel API avec rate limiting et retry exponentiel.

        Stratégie (config MISTRAL.max_retries, défaut 4) :
          Tentative 1 → délai 1s → Tentative 2 → délai 2s → … → 8s

        Args:
            model:       Identifiant du modèle Mistral.
            prompt:      Contenu utilisateur.
            system:      Prompt système.
            max_tokens:  Limite de tokens générés.
            temperature: Paramètre de créativité.
            use_cache:   Met en cache la réponse si True.

        Returns:
            MistralResponse rempli.
        """
        from config import MISTRAL

        last_error = ""
        delay      = MISTRAL.retry_base_delay

        for attempt in range(1, MISTRAL.max_retries + 1):
            self._throttle(MISTRAL.requests_per_sec)

            start_ts = time.perf_counter()
            try:
                response = self._http_post(
                    url     = f"{MISTRAL.base_url}/chat/completions",
                    api_key = MISTRAL.api_key,
                    payload = {
                        "model":       model,
                        "max_tokens":  max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": prompt},
                        ],
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                log.warning("Appel Mistral tentative %d/%d échouée : %s",
                            attempt, MISTRAL.max_retries, exc)
                if attempt < MISTRAL.max_retries:
                    log.debug("Retry dans %.0f s…", delay)
                    time.sleep(delay)
                    delay *= 2
                continue

            latency_ms = (time.perf_counter() - start_ts) * 1_000

            if response.get("error"):
                status      = response.get("_status_code", 0)
                last_error  = str(response["error"])
                if status == 429 or status >= 500:
                    log.warning("Erreur API %d (tentative %d/%d) : %s",
                                status, attempt, MISTRAL.max_retries, last_error)
                    if attempt < MISTRAL.max_retries:
                        time.sleep(delay)
                        delay *= 2
                    continue
                else:
                    log.error("Erreur API définitive %d : %s", status, last_error)
                    return MistralResponse(success=False, error=last_error,
                                           model=model, latency_ms=latency_ms)

            try:
                content    = response["choices"][0]["message"]["content"]
                usage      = response.get("usage", {})
                tokens_in  = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
            except (KeyError, IndexError) as exc:
                last_error = f"Réponse API mal formée : {exc}"
                log.error(last_error)
                return MistralResponse(success=False, error=last_error,
                                       model=model, latency_ms=latency_ms)

            resp = MistralResponse(
                success    = True,
                content    = content,
                model      = model,
                tokens_in  = tokens_in,
                tokens_out = tokens_out,
                latency_ms = round(latency_ms, 1),
            )
            self._record_success(model, tokens_in + tokens_out, latency_ms)
            self._check_quota_alert()
            if use_cache:
                self._put_cache(prompt, resp)

            log.debug("Codestral OK — %d tok in / %d tok out / %.0f ms",
                      tokens_in, tokens_out, latency_ms)
            return resp

        log.error("Mistral API inaccessible après %d tentatives. Mode dégradé activé.",
                  MISTRAL.max_retries)
        with self._lock:
            self._degraded = True
        self._push_metrics()
        return MistralResponse(
            success=False,
            error=f"Échec après {MISTRAL.max_retries} tentatives : {last_error}",
            model=model,
        )

    # ──────────────────────────────────────────────────────────────────
    # HTTP (urllib stdlib — zéro dépendance externe)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _http_post(url: str, api_key: str, payload: dict) -> dict:
        """
        Effectue un POST JSON via urllib (zéro dépendance externe).

        Args:
            url:     Endpoint complet de l'API.
            api_key: Clé d'authentification Bearer.
            payload: Corps de la requête.

        Returns:
            Dictionnaire JSON décodé avec "_status_code" injecté.

        Raises:
            OSError: Erreur réseau (timeout, DNS, etc.).
        """
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data    = data,
            method  = "POST",
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept":        "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body   = resp.read().decode("utf-8")
                result = json.loads(body)
                result["_status_code"] = resp.status
                return result
        except urllib.error.HTTPError as exc:
            try:
                err_data = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_data = {"message": exc.reason}
            err_data["_status_code"] = exc.code
            err_data["error"] = err_data.get("message", str(exc))
            return err_data
        except urllib.error.URLError as exc:
            raise OSError(f"Erreur réseau vers {url} : {exc.reason}") from exc

    # ──────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ──────────────────────────────────────────────────────────────────
    def _throttle(self, max_rps: float) -> None:
        """
        Attend le temps nécessaire pour respecter le rate limit.

        Args:
            max_rps: Requêtes par seconde autorisées.
        """
        min_interval = 1.0 / max_rps
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_call_ts = time.monotonic()

    # ──────────────────────────────────────────────────────────────────
    # CACHE MÉMOIRE
    # ──────────────────────────────────────────────────────────────────
    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _get_cached(self, prompt: str) -> MistralResponse | None:
        key = self._cache_key(prompt)
        with self._lock:
            return self._cache.get(key)

    def _put_cache(self, prompt: str, response: MistralResponse) -> None:
        key = self._cache_key(prompt)
        with self._lock:
            self._cache[key] = response

    # ──────────────────────────────────────────────────────────────────
    # COMPTABILITÉ ET MÉTRIQUES
    # ──────────────────────────────────────────────────────────────────
    def _record_success(self, model: str, tokens: int, latency_ms: float) -> None:
        with self._lock:
            self._usage.reset_if_new_month()
            self._usage.tokens += tokens
            self._call_stats[model] = self._call_stats.get(model, 0) + 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._MAX_LATENCY_HISTORY:
                self._latencies.pop(0)
        metrics.increment("model", "api_calls", flush=False)
        metrics.increment("model", "api_tokens_this_month", tokens, flush=False)
        self._push_metrics()

    def _push_metrics(self) -> None:
        try:
            metrics.update_module("mistral_client", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques Mistral : %s", exc)

    def _check_quota_alert(self) -> None:
        from config import MISTRAL
        with self._lock:
            used_pct = (
                self._usage.tokens / MISTRAL.monthly_token_budget
                if MISTRAL.monthly_token_budget else 0.0
            )
        if used_pct >= MISTRAL.alert_threshold_pct:
            log.warning(
                "⚠  Quota Mistral : %.1f%% utilisé ce mois (%s/%s tokens).",
                used_pct * 100,
                f"{self._usage.tokens:,}",
                f"{MISTRAL.monthly_token_budget:,}",
            )


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE SINGLETON
# ══════════════════════════════════════════════════════════════════════
mistral = MistralClient()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python api/mistral_client.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os

    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — api/mistral_client.py  smoke test")
    print("═" * 60)
    print()

    # Test 1 : refus sans clé API
    os.environ.pop("MISTRAL_API_KEY", None)
    resp = mistral.complete_code("def foo(): pass", '{"latency_ms": 0}')
    assert not resp.success, "Doit échouer sans clé API"
    assert "MISTRAL_API_KEY" in resp.error
    print("✅  Test 1 : refus correct sans MISTRAL_API_KEY")

    # Test 2 : structure des métriques
    m = mistral.get_metrics()
    assert isinstance(m["tokens_this_month"], int)
    assert isinstance(m["degraded_mode"], bool)
    assert isinstance(m["cache_entries"], int)
    print("✅  Test 2 : get_metrics() structure valide")

    # Test 3 : cache put/get/clear
    fake = MistralResponse(success=True, content='{"changes":[]}', model="test")
    mistral._put_cache("test-prompt", fake)
    assert mistral._get_cached("test-prompt") is not None
    assert mistral.get_metrics()["cache_entries"] == 1
    mistral.clear_cache()
    assert mistral.get_metrics()["cache_entries"] == 0
    print("✅  Test 3 : cache opérationnel")

    # Test 4 : mode dégradé
    mistral._degraded = True
    assert not mistral.is_available()
    mistral.reset_degraded()
    print("✅  Test 4 : mode dégradé et reset OK")

    # Test 5 : rate limiting
    start = time.perf_counter()
    mistral._throttle(1.0)
    mistral._throttle(1.0)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.9, f"Rate limiting insuffisant : {elapsed:.2f}s"
    print(f"✅  Test 5 : rate limiting OK ({elapsed:.2f}s)")

    print("\n✅  Tous les tests api/mistral_client.py sont passés.")