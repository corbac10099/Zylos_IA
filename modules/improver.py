"""
modules/improver.py — Auto-amélioration du code source de ZYLOS AI
====================================================================
Analyse les modules Python du projet via Mistral Codestral et propose
(ou applique automatiquement) des améliorations ciblées.

RÈGLE DE SÉCURITÉ ABSOLUE :
  Ce module est le SEUL autorisé à appeler l'API Mistral.
  Il opère exclusivement sur le code source du projet,
  jamais sur les réponses à l'utilisateur.

Pipeline d'un cycle :
  1. Sélection des modules à analyser (rotation, exclusions respectées)
  2. Lecture du code source + métriques du module
  3. Appel Mistral Codestral via api/mistral_client.py
  4. Parsing du JSON de suggestions
  5. Validation syntaxique (py_compile) avant toute application
  6. Sauvegarde dans data/backups/pending/ (auto_apply=False)
     ou application directe (auto_apply=True, DANGER)
  7. Backup automatique avant toute modification

Usage :
    from modules.improver import improver
    suggestions = improver.run_cycle()   # analyse N modules
    improver.apply_pending()             # applique les suggestions en attente
"""

from __future__ import annotations

import ast
import json
import py_compile
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Suggestion:
    """
    Suggestion d'amélioration retournée par Codestral.

    Attributes:
        module_path:   Chemin relatif du module concerné (ex: "modules/brain.py").
        analysis:      Analyse textuelle de Codestral.
        changes:       Liste de modifications proposées.
        priority:      "high" | "medium" | "low"
        estimated_improvement: Description de l'amélioration attendue.
        confidence:    Score de confiance [0.0, 1.0] calculé localement.
        raw_json:      JSON brut retourné par l'API.
        created_at:    Timestamp ISO de création.
        applied:       True si la suggestion a été appliquée.
        apply_error:   Message d'erreur si l'application a échoué.
    """
    module_path:            str
    analysis:               str
    changes:                list[dict[str, Any]]
    priority:               str   = "medium"
    estimated_improvement:  str   = ""
    confidence:             float = 0.0
    raw_json:               str   = ""
    created_at:             str   = ""
    applied:                bool  = False
    apply_error:            str   = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.confidence:
            self.confidence = self._compute_confidence()

    def _compute_confidence(self) -> float:
        """
        Calcule un score de confiance heuristique basé sur la structure
        de la réponse Codestral.
        """
        if not self.changes:
            return 0.0
        score = 0.5
        # Bonus si l'analyse est non vide
        if len(self.analysis) > 50:
            score += 0.1
        # Bonus par changement avec raison et code présents
        for change in self.changes:
            if change.get("reason") and change.get("new_code"):
                score += 0.1
        # Bonus priorité haute
        if self.priority == "high":
            score += 0.1
        return min(1.0, round(score, 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path":           self.module_path,
            "analysis":              self.analysis,
            "changes":               self.changes,
            "priority":              self.priority,
            "estimated_improvement": self.estimated_improvement,
            "confidence":            self.confidence,
            "raw_json":              self.raw_json,
            "created_at":            self.created_at,
            "applied":               self.applied,
            "apply_error":           self.apply_error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Suggestion":
        return cls(
            module_path           = d.get("module_path", ""),
            analysis              = d.get("analysis", ""),
            changes               = d.get("changes", []),
            priority              = d.get("priority", "medium"),
            estimated_improvement = d.get("estimated_improvement", ""),
            confidence            = d.get("confidence", 0.0),
            raw_json              = d.get("raw_json", ""),
            created_at            = d.get("created_at", ""),
            applied               = d.get("applied", False),
            apply_error           = d.get("apply_error", ""),
        )


# ══════════════════════════════════════════════════════════════════════
# MODULE IMPROVER
# ══════════════════════════════════════════════════════════════════════
class Improver:
    """
    Auto-amélioration du code source via Mistral Codestral.

    Singleton exposé via `improver` en bas de fichier.

    ╔══════════════════════════════════════════════════════╗
    ║  Ce module appelle Mistral. Il est le SEUL à le     ║
    ║  faire dans tout le projet ZYLOS AI.                ║
    ╚══════════════════════════════════════════════════════╝

    Attributes:
        _lock:            Verrou pour empêcher les cycles concurrents.
        _module_cursor:   Index de rotation pour la sélection des modules.
        _pending:         Suggestions en attente d'application.
        _stats:           Compteurs de cycles et de modifications.
    """

    def __init__(self) -> None:
        self._lock           = threading.Lock()
        self._module_cursor  = 0
        self._pending: list[Suggestion] = []
        self._stats: dict[str, Any] = {
            "cycles_run":       0,
            "changes_applied":  0,
            "changes_pending":  0,
            "changes_rejected": 0,
            "success_rate":     0.0,
        }

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def run_cycle(self, max_modules: int = 3) -> list[Suggestion]:
        """
        Exécute un cycle d'analyse : sélectionne et analyse N modules.

        Args:
            max_modules: Nombre maximum de modules à analyser par cycle.

        Returns:
            Liste de Suggestion générées (non encore appliquées si
            auto_apply=False).

        Example:
            >>> suggestions = improver.run_cycle(max_modules=2)
        """
        from config import IMPROVER, MISTRAL

        if not IMPROVER.enabled:
            log.info("Improver désactivé (IMPROVER.enabled=False).")
            return []

        if not MISTRAL.is_configured():
            log.warning("Improver : MISTRAL_API_KEY absente — cycle ignoré.")
            return []

        if not self._lock.acquire(blocking=False):
            log.warning("Improver : cycle déjà en cours.")
            return []

        try:
            return self._do_cycle(max_modules)
        finally:
            self._lock.release()

    def apply_pending(self) -> int:
        """
        Applique toutes les suggestions en attente (si auto_apply=False).

        Chaque modification est :
          1. Validée syntaxiquement (py_compile)
          2. Sauvegardée dans data/backups/
          3. Appliquée dans le code source

        Returns:
            Nombre de modifications effectivement appliquées.

        Example:
            >>> n = improver.apply_pending()
            >>> print(f"{n} modifications appliquées.")
        """
        self._load_pending()
        applied = 0

        for suggestion in list(self._pending):
            if suggestion.applied:
                continue
            n = self._apply_suggestion(suggestion)
            applied += n

        self._save_pending()
        self._update_stats()
        return applied

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques de l'improver."""
        with self._lock if False else _null_ctx():
            return {
                "cycles_run":       int(self._stats["cycles_run"]),
                "changes_applied":  int(self._stats["changes_applied"]),
                "changes_pending":  len([s for s in self._pending if not s.applied]),
                "changes_rejected": int(self._stats["changes_rejected"]),
                "success_rate":     float(self._stats["success_rate"]),
            }

    # ──────────────────────────────────────────────────────────────────
    # CYCLE INTERNE
    # ──────────────────────────────────────────────────────────────────
    def _do_cycle(self, max_modules: int) -> list[Suggestion]:
        """Logique interne d'un cycle (appelée sous verrou)."""
        from config import IMPROVER, PATHS

        modules = self._select_modules(max_modules, PATHS, IMPROVER)
        if not modules:
            log.info("Improver : aucun module à analyser.")
            return []

        suggestions: list[Suggestion] = []
        for mod_path in modules:
            log.info("Improver — analyse de %s …", mod_path.name)
            sug = self._analyze_module(mod_path)
            if sug is not None:
                suggestions.append(sug)

        # Filtrage par score de confiance minimum
        from config import IMPROVER as IMP
        good = [s for s in suggestions if s.confidence >= IMP.min_confidence_score]
        log.info("Improver : %d/%d suggestions retenues (confiance ≥ %.2f).",
                 len(good), len(suggestions), IMP.min_confidence_score)

        # Sauvegarde dans pending
        self._pending.extend(good)
        self._save_pending()

        # Auto-apply si configuré
        if IMP.auto_apply:
            log.warning("⚠  AUTO_APPLY activé — application immédiate des suggestions.")
            for sug in good:
                self._apply_suggestion(sug)

        # Métriques
        self._stats["cycles_run"] = int(self._stats["cycles_run"]) + 1
        metrics.update("improvements", {
            "cycles_run":      int(self._stats["cycles_run"]),
            "changes_pending": len(good),
        }, flush=True)
        self._push_metrics()

        return good

    # ──────────────────────────────────────────────────────────────────
    # SÉLECTION DES MODULES
    # ──────────────────────────────────────────────────────────────────
    def _select_modules(
        self,
        max_n:    int,
        paths:    Any,
        improver_cfg: Any,
    ) -> list[Path]:
        """
        Sélectionne les modules Python à analyser en rotation.
        Respecte les exclusions configurées.

        Returns:
            Liste de Path vers les fichiers .py à analyser.
        """
        excluded = set(improver_cfg.excluded_modules)

        # Collecter tous les .py du projet (hors __pycache__, venv, etc.)
        root      = paths.root
        all_py: list[Path] = []
        for pat in ["*.py", "*/*.py", "*/*/*.py"]:
            for p in sorted(root.glob(pat)):
                # Ignorer les dossiers exclus
                parts = p.parts
                if any(x in parts for x in ("__pycache__", ".venv", "venv", "ENV",
                                              "build", "dist", ".git")):
                    continue
                rel = str(p.relative_to(root))
                if rel in excluded:
                    continue
                if p.stat().st_size < 200:
                    continue
                all_py.append(p)

        if not all_py:
            return []

        # Rotation avec le curseur
        start   = self._module_cursor % len(all_py)
        selected: list[Path] = []
        for i in range(len(all_py)):
            idx = (start + i) % len(all_py)
            selected.append(all_py[idx])
            if len(selected) >= max_n:
                break

        self._module_cursor = (start + len(selected)) % len(all_py)
        return selected

    # ──────────────────────────────────────────────────────────────────
    # ANALYSE D'UN MODULE
    # ──────────────────────────────────────────────────────────────────
    def _analyze_module(self, mod_path: Path) -> Suggestion | None:
        """
        Lit le code source, récupère les métriques du module,
        et appelle Mistral Codestral pour l'analyse.

        Args:
            mod_path: Chemin absolu vers le fichier .py.

        Returns:
            Suggestion ou None si l'appel a échoué ou n'a rien produit.
        """
        from config import PATHS

        # Lecture du code source
        try:
            code = mod_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Improver : impossible de lire %s : %s", mod_path, exc)
            return None

        # Métriques du module (depuis le registre)
        rel_path    = str(mod_path.relative_to(PATHS.root))
        module_name = rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
        module_metrics = metrics.get_module(module_name.split(".")[-1])
        metrics_json   = json.dumps(module_metrics or {}, ensure_ascii=False, indent=2)

        # Contexte historique (métriques globales condensées)
        snap    = metrics.snapshot()
        context = (
            f"Sessions complétées : {snap.get('training', {}).get('sessions_completed', 0)}\n"
            f"Tokens générés : {snap.get('model', {}).get('tokens_generated', 0)}\n"
            f"RAG hit rate : {snap.get('memory', {}).get('rag_hit_rate', 0.0):.2%}"
        )

        # Appel Mistral Codestral
        from api.mistral_client import mistral
        t0   = time.perf_counter()
        resp = mistral.complete_code(code, metrics_json, context)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        if not resp.success:
            log.warning("Improver : Codestral a échoué pour %s : %s", rel_path, resp.error)
            return None

        # Parsing du JSON retourné
        return self._parse_response(resp.content, rel_path)

    # ──────────────────────────────────────────────────────────────────
    # PARSING DE LA RÉPONSE
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_response(content: str, module_path: str) -> Suggestion | None:
        """
        Parse la réponse JSON de Codestral en Suggestion.

        Gère les cas où le modèle ajoute des backticks ou du texte parasite.

        Args:
            content:     Réponse brute de Codestral.
            module_path: Chemin relatif du module (pour les logs).

        Returns:
            Suggestion ou None si le JSON est invalide.
        """
        # Nettoyage des backticks markdown éventuels
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.warning("Improver : JSON invalide pour %s : %s", module_path, exc)
            return None

        changes  = data.get("changes", [])
        analysis = data.get("analysis", "")

        if not isinstance(changes, list):
            changes = []

        # Pas de changements proposés → rien à faire
        if not changes:
            log.debug("Improver : aucun changement proposé pour %s.", module_path)
            return None

        return Suggestion(
            module_path           = module_path,
            analysis              = analysis,
            changes               = changes,
            priority              = data.get("priority", "medium"),
            estimated_improvement = data.get("estimated_improvement", ""),
            raw_json              = content[:2000],  # tronqué pour le stockage
        )

    # ──────────────────────────────────────────────────────────────────
    # APPLICATION DES SUGGESTIONS
    # ──────────────────────────────────────────────────────────────────
    def _apply_suggestion(self, suggestion: Suggestion) -> int:
        """
        Tente d'appliquer les changements d'une Suggestion.

        Pour chaque changement :
          1. Localise la fonction dans le fichier source
          2. Valide le nouveau code Python (py_compile + ast.parse)
          3. Crée un backup du fichier original
          4. Remplace la fonction dans le fichier source

        Args:
            suggestion: Suggestion à appliquer.

        Returns:
            Nombre de changements appliqués avec succès.
        """
        from config import PATHS

        mod_file = PATHS.root / suggestion.module_path
        if not mod_file.exists():
            suggestion.apply_error = f"Fichier introuvable : {mod_file}"
            return 0

        applied = 0
        for change in suggestion.changes:
            change_type = change.get("type", "")
            if change_type != "replace_function":
                log.debug("Improver : type de changement non supporté : %s", change_type)
                continue

            fn_name  = change.get("function_name", "")
            new_code = change.get("new_code", "")
            reason   = change.get("reason", "")

            if not fn_name or not new_code:
                continue

            # Validation syntaxique du nouveau code
            if not self._validate_python(new_code, fn_name):
                self._stats["changes_rejected"] = int(self._stats["changes_rejected"]) + 1
                log.warning("Improver : syntaxe invalide pour %s.%s — ignoré.",
                            suggestion.module_path, fn_name)
                continue

            # Backup du fichier original
            self._backup_file(mod_file, PATHS)

            # Remplacement
            ok = self._replace_function(mod_file, fn_name, new_code)
            if ok:
                applied += 1
                self._stats["changes_applied"] = int(self._stats["changes_applied"]) + 1
                log.info("Improver : ✅ %s.%s modifié (%s).",
                         suggestion.module_path, fn_name, reason[:60])
            else:
                self._stats["changes_rejected"] = int(self._stats["changes_rejected"]) + 1
                log.warning("Improver : impossible de remplacer %s.%s.",
                             suggestion.module_path, fn_name)

        suggestion.applied = applied > 0
        return applied

    @staticmethod
    def _validate_python(code: str, context: str = "") -> bool:
        """
        Valide que le code Python est syntaxiquement correct.

        Utilise à la fois ast.parse et py_compile sur un fichier temporaire.

        Args:
            code:    Code Python à valider.
            context: Nom de la fonction (pour les logs).

        Returns:
            True si le code est syntaxiquement valide.
        """
        # Test 1 : ast.parse
        try:
            ast.parse(code)
        except SyntaxError as exc:
            log.debug("ast.parse échoué pour %s : %s", context, exc)
            return False

        # Test 2 : py_compile sur fichier temporaire
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            py_compile.compile(tmp_path, doraise=True)
            Path(tmp_path).unlink(missing_ok=True)
            return True
        except py_compile.PyCompileError as exc:
            log.debug("py_compile échoué pour %s : %s", context, exc)
            return False
        except Exception as exc:
            log.debug("Validation Python inattendue pour %s : %s", context, exc)
            return False

    @staticmethod
    def _replace_function(file_path: Path, fn_name: str, new_code: str) -> bool:
        """
        Remplace le corps d'une fonction dans un fichier Python source.

        Utilise ast pour localiser la définition de la fonction et
        effectue un remplacement de la plage de lignes correspondante.

        Args:
            file_path: Fichier Python à modifier.
            fn_name:   Nom de la fonction à remplacer.
            new_code:  Nouveau code complet de la fonction (incluant `def`).

        Returns:
            True si le remplacement a réussi.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
            lines  = source.splitlines(keepends=True)

            tree = ast.parse(source)
            fn_node = None

            # Chercher la définition de la fonction (y compris méthodes de classe)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == fn_name:
                        fn_node = node
                        break

            if fn_node is None:
                log.debug("Improver : fonction '%s' introuvable dans %s.",
                          fn_name, file_path.name)
                return False

            start_line = fn_node.lineno - 1       # 0-based
            end_line   = fn_node.end_lineno        # 0-based exclusive

            # Détecter l'indentation de la définition originale
            original_def = lines[start_line]
            indent       = len(original_def) - len(original_def.lstrip())
            indent_str   = " " * indent

            # Ré-indenter le nouveau code si nécessaire
            new_lines = new_code.splitlines(keepends=True)
            if new_lines:
                first_non_empty = next(
                    (i for i, l in enumerate(new_lines) if l.strip()), 0
                )
                new_indent = len(new_lines[first_non_empty]) - len(new_lines[first_non_empty].lstrip())
                if new_indent != indent:
                    reindented = []
                    for line in new_lines:
                        stripped = line.lstrip()
                        if stripped:
                            cur_indent = len(line) - len(stripped)
                            new_ind    = indent + max(0, cur_indent - new_indent)
                            reindented.append(" " * new_ind + stripped)
                        else:
                            reindented.append(line)
                    new_lines = reindented

            # Assurer un saut de ligne final
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # Remplacement
            new_source_lines = lines[:start_line] + new_lines + lines[end_line:]
            new_source       = "".join(new_source_lines)

            # Validation finale du fichier complet modifié
            try:
                ast.parse(new_source)
            except SyntaxError as exc:
                log.warning("Improver : fichier résultant invalide après remplacement de %s : %s",
                            fn_name, exc)
                return False

            file_path.write_text(new_source, encoding="utf-8")
            return True

        except Exception as exc:
            log.error("Improver : _replace_function inattendu : %s", exc)
            return False

    # ──────────────────────────────────────────────────────────────────
    # BACKUP
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _backup_file(file_path: Path, paths: Any) -> None:
        """Crée une copie de sauvegarde horodatée du fichier."""
        from datetime import datetime, timezone
        try:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup   = paths.backups / f"{file_path.name}.{ts}.bak"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(file_path), str(backup))
            log.debug("Backup créé : %s", backup)
        except Exception as exc:
            log.warning("Impossible de créer le backup de %s : %s", file_path, exc)

    # ──────────────────────────────────────────────────────────────────
    # PERSISTANCE DES SUGGESTIONS EN ATTENTE
    # ──────────────────────────────────────────────────────────────────
    def _load_pending(self) -> None:
        """Charge les suggestions en attente depuis data/backups/pending/."""
        from config import IMPROVER
        pending_dir = IMPROVER.pending_dir
        if not pending_dir.exists():
            return

        loaded: list[Suggestion] = []
        for jf in sorted(pending_dir.glob("suggestion_*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                loaded.append(Suggestion.from_dict(data))
            except Exception as exc:
                log.debug("Suggestion JSON invalide %s : %s", jf.name, exc)

        self._pending = loaded
        log.debug("Improver : %d suggestions en attente chargées.", len(loaded))

    def _save_pending(self) -> None:
        """Sauvegarde les suggestions en attente dans data/backups/pending/."""
        from config import IMPROVER
        pending_dir = IMPROVER.pending_dir
        try:
            pending_dir.mkdir(parents=True, exist_ok=True)
            # Réécrire toutes les suggestions (approche simple)
            for existing in pending_dir.glob("suggestion_*.json"):
                existing.unlink(missing_ok=True)
            for i, sug in enumerate(self._pending):
                path = pending_dir / f"suggestion_{i:04d}.json"
                path.write_text(
                    json.dumps(sug.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            log.error("Impossible de sauvegarder les suggestions : %s", exc)

    def _update_stats(self) -> None:
        applied  = sum(1 for s in self._pending if s.applied)
        total    = len(self._pending)
        rate     = applied / total if total else 0.0
        self._stats["success_rate"] = round(rate, 4)

    def _push_metrics(self) -> None:
        try:
            m = self.get_metrics()
            metrics.update("improvements", {
                "changes_applied": m["changes_applied"],
                "changes_pending": m["changes_pending"],
                "success_rate":    m["success_rate"],
            }, flush=False)
            metrics.update_module("improver", m, flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques improver : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRE
# ══════════════════════════════════════════════════════════════════════
class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
improver = Improver()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python modules/improver.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — modules/improver.py  smoke test")
    print("═" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        import config as cfg
        object.__setattr__(cfg.PATHS,    "root",    p)
        object.__setattr__(cfg.PATHS,    "backups", p / "backups")
        object.__setattr__(cfg.PATHS,    "logs",    p / "logs")
        object.__setattr__(cfg.IMPROVER, "pending_dir", p / "backups" / "pending")

        imp = Improver()

        # Test 1 : cycle sans clé API
        os.environ.pop("MISTRAL_API_KEY", None)
        sug = imp.run_cycle()
        assert sug == [], f"Sans clé API, doit retourner [] : {sug}"
        print("✅  Test 1 : refus sans clé API OK")

        # Test 2 : _validate_python
        good_code = "def foo(x):\n    return x * 2\n"
        bad_code  = "def foo(x\n    return x"
        assert Improver._validate_python(good_code, "foo")
        assert not Improver._validate_python(bad_code, "foo")
        print("✅  Test 2 : _validate_python OK")

        # Test 3 : _parse_response — JSON valide
        fake_json = json.dumps({
            "analysis":   "La fonction foo peut être optimisée.",
            "changes":    [{"type": "replace_function", "function_name": "foo",
                            "reason": "Optimisation", "new_code": "def foo(x):\n    return x + 1\n"}],
            "priority":   "medium",
            "estimated_improvement": "+5% vitesse",
        })
        sug2 = Improver._parse_response(fake_json, "test_module.py")
        assert sug2 is not None
        assert len(sug2.changes) == 1
        assert sug2.confidence > 0
        print(f"✅  Test 3 : _parse_response OK (confiance={sug2.confidence})")

        # Test 4 : _parse_response — JSON vide (changes=[])
        empty_json = '{"analysis": "Rien à changer.", "changes": [], "priority": "low"}'
        sug3 = Improver._parse_response(empty_json, "test.py")
        assert sug3 is None
        print("✅  Test 4 : _parse_response changes=[] → None OK")

        # Test 5 : _replace_function sur un vrai fichier
        test_py = p / "test_replace.py"
        test_py.write_text(
            'def foo(x):\n    return x\n\ndef bar(y):\n    return y + 1\n',
            encoding="utf-8",
        )
        new_fn = "def foo(x):\n    return x * 10\n"
        ok = Improver._replace_function(test_py, "foo", new_fn)
        assert ok, "Remplacement de foo doit réussir"
        result_code = test_py.read_text(encoding="utf-8")
        assert "x * 10" in result_code, "Nouveau code non présent"
        assert "bar" in result_code,    "bar ne doit pas être supprimée"
        print("✅  Test 5 : _replace_function OK")

        # Test 6 : _replace_function — fonction inexistante
        ok2 = Improver._replace_function(test_py, "nonexistent_fn", new_fn)
        assert not ok2
        print("✅  Test 6 : fonction inexistante → False OK")

        # Test 7 : métriques
        m = imp.get_metrics()
        assert isinstance(m["cycles_run"], int)
        assert isinstance(m["success_rate"], float)
        print(f"✅  Test 7 : get_metrics() OK — {m}")

    print("\n✅  Tous les tests modules/improver.py sont passés.")