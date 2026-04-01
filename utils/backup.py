"""
utils/backup.py — Système de sauvegarde et restauration de ZYLOS AI
====================================================================
Fournit des utilitaires de backup pour :
  - Les poids du modèle RWKV (fichiers .pth)
  - Les poids LoRA fine-tunés
  - L'état RNN persistant
  - La configuration et les métriques
  - Le corpus d'entraînement

RÈGLE : Ce module ne doit JAMAIS être soumis à l'auto-amélioration
        (listé dans config.IMPROVER.excluded_modules).

Usage :
    from utils.backup import backup
    archive = backup.create_snapshot("avant_training")
    backup.restore_snapshot(archive)
    backup.cleanup_old(keep_n=10)
"""

from __future__ import annotations

import json
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# GESTIONNAIRE DE BACKUPS
# ══════════════════════════════════════════════════════════════════════
class BackupManager:
    """
    Gestionnaire centralisé des sauvegardes de ZYLOS AI.

    Chaque snapshot est un fichier ZIP horodaté contenant :
      - data/metrics.json
      - data/lora_weights/ (le dernier dossier LoRA)
      - data/rwkv_state.pt (état RNN)
      - Un manifest.json décrivant le contenu

    Attributes:
        _lock: Verrou pour empêcher les backups concurrents.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # SNAPSHOTS COMPLETS
    # ──────────────────────────────────────────────────────────────────
    def create_snapshot(self, label: str = "") -> Path | None:
        """
        Crée un snapshot complet de l'état courant dans data/backups/.

        Le snapshot inclut : métriques, poids LoRA, état RNN.
        Les gros fichiers .pth du modèle de base ne sont pas copiés
        (ils peuvent être re-téléchargés depuis HuggingFace).

        Args:
            label: Libellé optionnel ajouté au nom du fichier.

        Returns:
            Chemin vers le fichier ZIP créé, ou None en cas d'erreur.

        Example:
            >>> archive = backup.create_snapshot("pre_training")
            >>> print(archive)  # data/backups/snapshot_20250714_030000_pre_training.zip
        """
        from config import PATHS

        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        label_s  = f"_{label}" if label else ""
        zip_name = f"snapshot_{ts}{label_s}.zip"
        zip_path = PATHS.backups / zip_name

        with self._lock:
            try:
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "label":      label,
                    "files":      [],
                }

                with zipfile.ZipFile(str(zip_path), "w",
                                     compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=6) as zf:

                    # metrics.json
                    if PATHS.metrics_file.exists():
                        zf.write(str(PATHS.metrics_file),
                                 arcname="metrics.json")
                        manifest["files"].append("metrics.json")

                    # État RNN
                    rwkv_state = PATHS.data / "rwkv_state.pt"
                    if rwkv_state.exists():
                        zf.write(str(rwkv_state), arcname="rwkv_state.pt")
                        manifest["files"].append("rwkv_state.pt")

                    # Dernier dossier LoRA
                    lora_dirs = sorted(PATHS.lora_weights.glob("lora_*")) if PATHS.lora_weights.exists() else []
                    if lora_dirs:
                        latest = lora_dirs[-1]
                        for lora_file in latest.rglob("*"):
                            if lora_file.is_file():
                                arcname = f"lora/{lora_file.relative_to(PATHS.lora_weights)}"
                                zf.write(str(lora_file), arcname=arcname)
                                manifest["files"].append(arcname)

                    # Manifest
                    zf.writestr("manifest.json",
                                json.dumps(manifest, ensure_ascii=False, indent=2))

                size_mb = zip_path.stat().st_size // (1024 ** 2)
                log.info("Snapshot créé : %s (%d Mo, %d fichiers).",
                         zip_name, size_mb, len(manifest["files"]))
                return zip_path

            except Exception as exc:
                log.error("Impossible de créer le snapshot : %s", exc)
                if zip_path.exists():
                    zip_path.unlink(missing_ok=True)
                return None

    def restore_snapshot(self, zip_path: Path) -> bool:
        """
        Restaure un snapshot complet depuis un fichier ZIP.

        Les fichiers sont extraits dans data/ en écrasant l'existant.
        Un backup de l'état courant est créé avant la restauration.

        Args:
            zip_path: Chemin vers le ZIP de snapshot à restaurer.

        Returns:
            True si la restauration a réussi.

        Example:
            >>> backup.restore_snapshot(Path("data/backups/snapshot_20250714_030000.zip"))
        """
        if not zip_path.exists():
            log.error("Snapshot introuvable : %s", zip_path)
            return False

        from config import PATHS

        # Backup de sécurité avant restauration
        self.create_snapshot("pre_restore")

        with self._lock:
            try:
                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    # Vérifier le manifest
                    try:
                        manifest = json.loads(zf.read("manifest.json"))
                    except Exception:
                        manifest = {"files": []}

                    log.info("Restauration de %s (%d fichiers)…",
                             zip_path.name, len(manifest.get("files", [])))

                    # Extraire dans data/
                    zf.extractall(str(PATHS.data))

                    # Déplacer les poids LoRA si présents
                    extracted_lora = PATHS.data / "lora"
                    if extracted_lora.exists():
                        target = PATHS.lora_weights / "lora_restored"
                        shutil.move(str(extracted_lora), str(target))

                log.info("Snapshot restauré depuis %s.", zip_path.name)
                return True

            except Exception as exc:
                log.error("Erreur lors de la restauration de %s : %s", zip_path.name, exc)
                return False

    # ──────────────────────────────────────────────────────────────────
    # BACKUP D'UN FICHIER UNIQUE
    # ──────────────────────────────────────────────────────────────────
    def backup_file(self, file_path: Path, label: str = "") -> Path | None:
        """
        Crée une copie horodatée d'un fichier unique dans data/backups/.

        Utile pour les sauvegardes pré-modification des modules Python.

        Args:
            file_path: Fichier à sauvegarder.
            label:     Libellé optionnel pour le nom du backup.

        Returns:
            Chemin vers la copie, ou None en cas d'erreur.

        Example:
            >>> bak = backup.backup_file(Path("modules/brain.py"), "pre_improver")
        """
        from config import PATHS

        if not file_path.exists():
            log.warning("backup_file : fichier introuvable : %s", file_path)
            return None

        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        label_s  = f"_{label}" if label else ""
        bak_name = f"{file_path.name}.{ts}{label_s}.bak"
        bak_path = PATHS.backups / bak_name

        try:
            PATHS.backups.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(file_path), str(bak_path))
            log.debug("Backup fichier : %s → %s", file_path.name, bak_name)
            return bak_path
        except Exception as exc:
            log.warning("backup_file échoué pour %s : %s", file_path, exc)
            return None

    def restore_file(self, bak_path: Path, dest: Path | None = None) -> bool:
        """
        Restaure un fichier depuis un backup .bak.

        Args:
            bak_path: Fichier .bak à restaurer.
            dest:     Destination (défaut : déduit du nom du .bak).

        Returns:
            True si la restauration a réussi.
        """
        if not bak_path.exists():
            log.error("restore_file : backup introuvable : %s", bak_path)
            return False

        if dest is None:
            # Reconstruire le nom original : module.py.20250714_030000_label.bak
            # → module.py
            name = bak_path.name
            # Supprimer le suffixe .bak et les composantes timestamp/label
            parts = name.split(".")
            if len(parts) >= 3 and parts[-1] == "bak":
                original_name = ".".join(parts[:-2])  # retire ts.bak
                from config import PATHS
                dest = PATHS.root / original_name
            else:
                log.error("Impossible de déduire la destination depuis %s", bak_path.name)
                return False

        try:
            shutil.copy2(str(bak_path), str(dest))
            log.info("Fichier restauré : %s → %s", bak_path.name, dest)
            return True
        except Exception as exc:
            log.error("restore_file échoué : %s", exc)
            return False

    # ──────────────────────────────────────────────────────────────────
    # NETTOYAGE
    # ──────────────────────────────────────────────────────────────────
    def cleanup_old(self, keep_n: int = 10) -> int:
        """
        Supprime les anciens snapshots en gardant les N plus récents.
        Les fichiers .bak individuels datant de plus de 30 jours sont
        également supprimés.

        Args:
            keep_n: Nombre de snapshots à conserver.

        Returns:
            Nombre de fichiers supprimés.

        Example:
            >>> n = backup.cleanup_old(keep_n=10)
            >>> print(f"{n} fichiers supprimés.")
        """
        from config import PATHS

        if not PATHS.backups.exists():
            return 0

        deleted = 0

        # Nettoyage des snapshots ZIP
        snapshots = sorted(PATHS.backups.glob("snapshot_*.zip"),
                           key=lambda p: p.stat().st_mtime)
        to_delete = snapshots[:-keep_n] if len(snapshots) > keep_n else []
        for f in to_delete:
            try:
                f.unlink()
                deleted += 1
                log.debug("Supprimé : %s", f.name)
            except Exception:
                pass

        # Nettoyage des .bak anciens (> 30 jours)
        import time
        cutoff = time.time() - 30 * 86400
        for bak in PATHS.backups.glob("*.bak"):
            try:
                if bak.stat().st_mtime < cutoff:
                    bak.unlink()
                    deleted += 1
            except Exception:
                pass

        if deleted:
            log.info("Cleanup backups : %d fichiers supprimés.", deleted)
        return deleted

    def list_snapshots(self) -> list[dict[str, Any]]:
        """
        Liste tous les snapshots disponibles avec leurs métadonnées.

        Returns:
            Liste de dicts : {"path", "name", "size_mb", "created_at"}.
        """
        from config import PATHS

        if not PATHS.backups.exists():
            return []

        result = []
        for zf in sorted(PATHS.backups.glob("snapshot_*.zip"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            info: dict[str, Any] = {
                "path":       str(zf),
                "name":       zf.name,
                "size_mb":    zf.stat().st_size // (1024 ** 2),
                "created_at": datetime.fromtimestamp(
                    zf.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "manifest":   {},
            }
            try:
                with zipfile.ZipFile(str(zf), "r") as z:
                    info["manifest"] = json.loads(z.read("manifest.json"))
            except Exception:
                pass
            result.append(info)

        return result


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
backup = BackupManager()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python utils/backup.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — utils/backup.py  smoke test")
    print("═" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        import config as cfg
        object.__setattr__(cfg.PATHS, "data",         p / "data")
        object.__setattr__(cfg.PATHS, "backups",      p / "data" / "backups")
        object.__setattr__(cfg.PATHS, "lora_weights", p / "data" / "lora_weights")
        object.__setattr__(cfg.PATHS, "metrics_file", p / "data" / "metrics.json")
        object.__setattr__(cfg.PATHS, "root",         p)

        bm = BackupManager()

        # Créer quelques fichiers factices
        (p / "data").mkdir(parents=True)
        (p / "data" / "metrics.json").write_text('{"test": true}')
        (p / "data" / "rwkv_state.pt").write_text("fake state")
        lora_dir = p / "data" / "lora_weights" / "lora_20250714_030000"
        lora_dir.mkdir(parents=True)
        (lora_dir / "lora_weights.pt").write_text("fake lora")

        # Test 1 : création de snapshot
        snap = bm.create_snapshot("test")
        assert snap is not None and snap.exists(), "Snapshot non créé"
        assert snap.suffix == ".zip"
        print(f"✅  Test 1 : create_snapshot OK — {snap.name}")

        # Test 2 : manifest dans le ZIP
        with zipfile.ZipFile(str(snap), "r") as zf:
            names    = zf.namelist()
            manifest = json.loads(zf.read("manifest.json"))
        assert "manifest.json" in names
        assert "metrics.json"  in names
        assert len(manifest["files"]) > 0
        print(f"✅  Test 2 : manifest ZIP OK — fichiers : {manifest['files']}")

        # Test 3 : list_snapshots
        snaps = bm.list_snapshots()
        assert len(snaps) >= 1
        assert snaps[0]["size_mb"] >= 0
        print(f"✅  Test 3 : list_snapshots OK ({len(snaps)} snapshot(s))")

        # Test 4 : backup d'un fichier unique
        test_file = p / "test_module.py"
        test_file.write_text("def hello(): pass\n")
        bak = bm.backup_file(test_file, "pre_test")
        assert bak is not None and bak.exists()
        print(f"✅  Test 4 : backup_file OK — {bak.name}")

        # Test 5 : cleanup_old
        # Créer plusieurs snapshots
        for i in range(5):
            bm.create_snapshot(f"snap_{i}")
        all_snaps = bm.list_snapshots()
        deleted = bm.cleanup_old(keep_n=3)
        remaining = bm.list_snapshots()
        assert len(remaining) <= 3, f"Trop de snapshots restants : {len(remaining)}"
        print(f"✅  Test 5 : cleanup_old OK ({deleted} supprimés, {len(remaining)} restants)")

        # Test 6 : restore_snapshot
        snap2 = bm.create_snapshot("restore_test")
        assert snap2 is not None
        ok = bm.restore_snapshot(snap2)
        assert ok
        print(f"✅  Test 6 : restore_snapshot OK")

        # Test 7 : backup fichier inexistant
        bak2 = bm.backup_file(p / "nonexistent.py")
        assert bak2 is None
        print("✅  Test 7 : backup fichier inexistant → None OK")

    print("\n✅  Tous les tests utils/backup.py sont passés.")