"""
modules/brain.py — Cerveau de ZYLOS AI
=======================================
Orchestre la génération de réponses en combinant :
  - RWKV local (seul et unique moteur de réponse utilisateur)
  - Mémoire vectorielle (RAG via modules/vectordb.py)
  - Historique de conversation fenêtré
  - Chain-of-thought optionnel pour améliorer le raisonnement

Mistral N'EST PAS utilisé ici. Ce module est 100% local.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Turn:
    role:    str
    content: str


@dataclass
class ChatResponse:
    """
    Résultat d'un appel à brain.chat().

    Attributes:
        text:         Réponse générée par RWKV.
        rag_entries:  Chunks mémoire utilisés.
        tokens_out:   Nombre de tokens générés (estimation).
        latency_ms:   Durée totale de génération.
        rag_hit:      True si au moins un chunk pertinent a été trouvé.
        error:        Message d'erreur si la génération a échoué.
        reasoning:    Étape de raisonnement intermédiaire (si chain-of-thought).
    """
    text:        str
    rag_entries: list[Any] = field(default_factory=list)
    tokens_out:  int       = 0
    latency_ms:  float     = 0.0
    rag_hit:     bool      = False
    error:       str       = ""
    reasoning:   str       = ""

    @property
    def success(self) -> bool:
        return not self.error and bool(self.text)


# ══════════════════════════════════════════════════════════════════════
# CERVEAU PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class Brain:
    """
    Interface de haut niveau pour la génération de réponses ZYLOS.

    Améliorations v2 :
      - Chain-of-thought : le modèle réfléchit avant de répondre
      - Détection de complexité : active CoT sur les questions difficiles
      - Nettoyage des balises de raisonnement avant retour utilisateur
    """

    def __init__(self) -> None:
        self._model:   Any        = None
        self._history: list[Turn] = []
        self._lock                = threading.Lock()
        self._stats: dict[str, Any] = {
            "total_calls":  0,
            "rag_hits":     0,
            "tokens_out":   0,
            "latencies_ms": [],
        }

    # ──────────────────────────────────────────────────────────────────
    # INJECTION DU MODÈLE
    # ──────────────────────────────────────────────────────────────────
    def set_model(self, rwkv_model: Any) -> None:
        self._model = rwkv_model
        log.info("Brain : modèle RWKV injecté.")

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def chat(
        self,
        user_message: str,
        max_tokens:   int | None = None,
        use_rag:      bool       = True,
    ) -> ChatResponse:
        if not user_message.strip():
            return ChatResponse(text="", error="Message vide.")

        from config import RWKV as RWKV_CFG, BRAIN

        t0       = time.perf_counter()
        n_tokens = max_tokens or RWKV_CFG.max_tokens

        rag_entries: list[Any] = []
        if use_rag:
            try:
                from modules.vectordb import vectordb
                rag_entries = vectordb.search(user_message)
            except Exception as exc:
                log.warning("RAG search échouée : %s", exc)

        prompt = self._build_prompt(user_message, rag_entries)

        reasoning = ""
        text      = ""
        error     = ""

        if self._model is None:
            text = self._degraded_response(user_message)
        else:
            try:
                # Chain-of-thought sur les questions complexes
                if BRAIN.chain_of_thought and self._needs_reasoning(user_message):
                    text, reasoning = self._generate_with_cot(prompt, n_tokens, RWKV_CFG)
                else:
                    text = self._model.generate(prompt, max_tokens=n_tokens)
            except Exception as exc:
                log.error("Erreur RWKV generate : %s", exc)
                error = str(exc)

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        tokens_out = max(1, len(text) // 4)

        if text and not error:
            self._append_history(user_message, text)

        self._record(tokens_out, elapsed_ms, bool(rag_entries))
        metrics.increment("session", "interactions",    flush=False)
        metrics.increment("model",   "rwkv_calls",      flush=False)
        metrics.increment("model",   "tokens_generated", tokens_out, flush=False)
        self._push_metrics()

        return ChatResponse(
            text        = text,
            rag_entries = rag_entries,
            tokens_out  = tokens_out,
            latency_ms  = round(elapsed_ms, 1),
            rag_hit     = bool(rag_entries),
            error       = error,
            reasoning   = reasoning,
        )

    def stream(
        self,
        user_message: str,
        max_tokens:   int | None  = None,
        use_rag:      bool        = True,
        on_token:     Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        from config import RWKV as RWKV_CFG

        n_tokens    = max_tokens or RWKV_CFG.max_tokens
        rag_entries: list[Any] = []

        if use_rag:
            try:
                from modules.vectordb import vectordb
                rag_entries = vectordb.search(user_message)
            except Exception as exc:
                log.warning("RAG search échouée (stream) : %s", exc)

        prompt    = self._build_prompt(user_message, rag_entries)
        full_text = ""
        t0        = time.perf_counter()

        if self._model is None:
            fallback = self._degraded_response(user_message)
            if on_token:
                on_token(fallback)
            yield fallback
            return

        # En mode streaming, on filtre les balises <think>...</think>
        # pour ne pas les afficher à l'utilisateur
        try:
            in_think = False
            buf      = ""
            for token in self._model.stream(prompt, max_tokens=n_tokens):
                full_text += token
                buf       += token

                # Filtrer les blocs de raisonnement <think>
                if "<think>" in buf:
                    in_think = True
                    buf      = buf[buf.index("<think>") + 7:]
                    continue
                if in_think:
                    if "</think>" in buf:
                        in_think = False
                        buf      = buf[buf.index("</think>") + 9:]
                        # yield le contenu après </think>
                        if buf:
                            if on_token:
                                on_token(buf)
                            yield buf
                            buf = ""
                    continue

                if on_token:
                    on_token(token)
                yield token

        except Exception as exc:
            log.error("Erreur RWKV stream : %s", exc)
            return

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        tokens_out = max(1, len(full_text) // 4)

        if full_text:
            # Nettoyer les balises de raisonnement dans l'historique
            clean_text = re.sub(r"<think>.*?</think>", "", full_text,
                                flags=re.DOTALL).strip()
            self._append_history(user_message, clean_text or full_text)

        self._record(tokens_out, elapsed_ms, bool(rag_entries))
        metrics.increment("session", "interactions",    flush=False)
        metrics.increment("model",   "rwkv_calls",      flush=False)
        metrics.increment("model",   "tokens_generated", tokens_out, flush=False)
        self._push_metrics()

    def reset_history(self) -> None:
        with self._lock:
            self._history.clear()
        log.info("Historique de conversation réinitialisé.")

    def get_history(self) -> list[dict[str, str]]:
        with self._lock:
            return [{"role": t.role, "content": t.content} for t in self._history]

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            calls = int(self._stats["total_calls"])
            hits  = int(self._stats["rag_hits"])
            toks  = int(self._stats["tokens_out"])
            lats  = list(self._stats["latencies_ms"])
        avg_ms   = sum(lats) / len(lats) if lats else 0.0
        hit_rate = hits / calls if calls else 0.0
        avg_tps  = (toks / (sum(lats) / 1_000)) if lats and sum(lats) > 0 else 0.0
        return {
            "total_calls":      calls,
            "rag_hit_rate":     round(hit_rate, 4),
            "tokens_generated": toks,
            "avg_latency_ms":   round(avg_ms, 1),
            "avg_tokens_per_s": round(avg_tps, 1),
            "history_turns":    len(self._history),
            "model_ready":      self._model is not None,
        }

    # ──────────────────────────────────────────────────────────────────
    # CHAIN-OF-THOUGHT
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _needs_reasoning(message: str) -> bool:
        """
        Détecte si la question nécessite un raisonnement approfondi.
        Activé pour : code, maths, comparaisons, explications complexes, "pourquoi".
        """
        triggers = [
            "pourquoi", "comment", "explique", "analyse", "compare",
            "calcul", "math", "code", "programme", "algorithme", "erreur",
            "bug", "optimise", "debug", "def ", "class ", "import ",
            "résous", "résoudre", "problème", "étape", "plan",
        ]
        msg_lower = message.lower()
        return any(t in msg_lower for t in triggers) or len(message) > 120

    def _generate_with_cot(
        self, prompt: str, n_tokens: int, cfg: Any,
    ) -> tuple[str, str]:
        """
        Génère une réponse avec chain-of-thought.
        Le modèle raisonne d'abord dans <think>...</think>,
        puis donne sa réponse finale.
        Retourne (réponse_nettoyée, raisonnement).
        """
        # Prompt enrichi pour guider le raisonnement
        cot_prompt = prompt.rstrip("Assistant:").rstrip() + (
            "\n\nAssistant: <think>\nRaisonnons étape par étape :\n"
        )

        raw = self._model.generate(cot_prompt, max_tokens=n_tokens + 256)

        # Extraction du raisonnement et de la réponse
        reasoning = ""
        answer    = raw

        think_match = re.search(r"<think>(.*?)</think>(.*)", raw, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            answer    = think_match.group(2).strip()
        else:
            # Le modèle a peut-être continué sans fermer <think>
            # → prendre tout comme réponse
            answer = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
            if not answer:
                answer = raw

        return answer, reasoning

    # ──────────────────────────────────────────────────────────────────
    # CONSTRUCTION DU PROMPT
    # ──────────────────────────────────────────────────────────────────
    def _build_prompt(self, user_message: str, rag_entries: list[Any]) -> str:
        """
        Construit le prompt complet : [SYSTEM] + [MÉMOIRE] + [HISTORIQUE] + [USER]
        Structure RWKV World optimisée pour le raisonnement.
        """
        from config import BRAIN, RWKV as RWKV_CFG

        parts: list[str] = []

        # ── Prompt système ────────────────────────────────────────────
        parts.append(f"System: {BRAIN.system_prompt}\n")

        # ── Mémoire RAG ───────────────────────────────────────────────
        if rag_entries:
            try:
                from modules.vectordb import vectordb
                context = vectordb.format_context(rag_entries)
                if context:
                    parts.append(f"Mémoire pertinente :\n{context}\n")
            except Exception as exc:
                log.debug("format_context échoué : %s", exc)

        # ── Historique fenêtré ────────────────────────────────────────
        history_block = self._get_windowed_history(
            BRAIN.max_history_turns,
            BRAIN.max_history_tokens,
        )
        if history_block:
            parts.append(history_block)

        # ── Message utilisateur ───────────────────────────────────────
        parts.append(f"User: {user_message}\n\nAssistant:")

        return "\n".join(parts)

    def _get_windowed_history(self, max_turns: int, max_tokens: int) -> str:
        with self._lock:
            history = list(self._history)
        if not history:
            return ""
        recent       = history[-(max_turns * 2):]
        lines:       list[str] = []
        total_chars  = 0
        budget_chars = max_tokens * 4

        for turn in reversed(recent):
            line = f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
            if total_chars + len(line) > budget_chars:
                break
            lines.insert(0, line)
            total_chars += len(line)

        return "\n".join(lines) + "\n" if lines else ""

    # ──────────────────────────────────────────────────────────────────
    # GESTION DE L'HISTORIQUE
    # ──────────────────────────────────────────────────────────────────
    def _append_history(self, user_msg: str, assistant_msg: str) -> None:
        from config import BRAIN
        max_msgs = BRAIN.max_history_turns * 2
        with self._lock:
            self._history.append(Turn(role="user",      content=user_msg))
            self._history.append(Turn(role="assistant", content=assistant_msg))
            while len(self._history) > max_msgs:
                self._history.pop(0)

    # ──────────────────────────────────────────────────────────────────
    # MODE DÉGRADÉ
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _degraded_response(user_message: str) -> str:
        return (
            "⏳ Le modèle RWKV est en cours de chargement. "
            "Veuillez patienter quelques instants avant de réessayer.\n"
            + (f"(Votre question : '{user_message[:80]}…')"
               if len(user_message) > 80
               else f"(Votre question : '{user_message}')")
        )

    # ──────────────────────────────────────────────────────────────────
    # COMPTABILITÉ
    # ──────────────────────────────────────────────────────────────────
    def _record(self, tokens_out: int, elapsed_ms: float, rag_hit: bool) -> None:
        with self._lock:
            self._stats["total_calls"] = int(self._stats["total_calls"]) + 1
            self._stats["tokens_out"]  = int(self._stats["tokens_out"]) + tokens_out
            if rag_hit:
                self._stats["rag_hits"] = int(self._stats["rag_hits"]) + 1
            lats = self._stats["latencies_ms"]
            if isinstance(lats, list):
                lats.append(elapsed_ms)
                if len(lats) > 100:
                    lats.pop(0)

    def _push_metrics(self) -> None:
        try:
            m = self.get_metrics()
            metrics.update("model", {"avg_tokens_per_sec": m["avg_tokens_per_s"]}, flush=False)
            metrics.update_module("brain", m, flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques brain : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
brain = Brain()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — modules/brain.py  smoke test")
    print("═" * 60)
    print()

    resp = brain.chat("Bonjour !", use_rag=False)
    assert not resp.error
    print(f"✅  Test 1 : mode dégradé OK — '{resp.text[:60]}'")

    tokens = list(brain.stream("Test stream", use_rag=False))
    assert len(tokens) > 0
    print(f"✅  Test 2 : stream dégradé OK")

    brain.reset_history()
    assert brain.get_history() == []
    brain._append_history("Bonjour", "Bonjour, comment puis-je vous aider ?")
    h = brain.get_history()
    assert len(h) == 2
    brain.reset_history()
    print("✅  Test 3 : historique OK")

    # Test _needs_reasoning
    assert brain._needs_reasoning("Comment fonctionne Python ?")
    assert brain._needs_reasoning("Explique-moi les transformers")
    assert not brain._needs_reasoning("Salut")
    print("✅  Test 4 : _needs_reasoning OK")

    # Test prompt
    prompt = brain._build_prompt("Qu'est-ce que RWKV ?", [])
    assert "System:" in prompt
    assert "User:"   in prompt
    assert "RWKV"    in prompt
    print("✅  Test 5 : _build_prompt OK")

    m = brain.get_metrics()
    assert isinstance(m["model_ready"], bool)
    print(f"✅  Test 6 : get_metrics OK")

    class _MockRWKV:
        def generate(self, prompt, max_tokens=768):
            return f"[MOCK] Réponse pour : {prompt[-30:]}"
        def stream(self, prompt, max_tokens=768):
            yield "[MOCK]"
            yield " réponse"
        def get_embeddings(self, text):
            return [0.1] * 512

    brain.set_model(_MockRWKV())
    resp = brain.chat("Test avec mock", use_rag=False)
    assert resp.success
    assert "[MOCK]" in resp.text
    print(f"✅  Test 7 : mock RWKV OK — '{resp.text[:50]}'")

    print("\n✅  Tous les tests modules/brain.py sont passés.")