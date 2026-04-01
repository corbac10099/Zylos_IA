"""
core/backend.py — Détection et initialisation automatique du backend GPU
=========================================================================
Détecte le meilleur backend disponible selon l'ordre de priorité :
  1. ROCm      — AMD sur Linux  (PyTorch + HIP, device "cuda" via HIP)
  2. DirectML  — AMD sur Windows (torch-directml)
  3. CUDA      — NVIDIA         (PyTorch standard)
  4. CPU       — fallback universel

Le backend est sélectionné au premier import et mis en cache dans
l'instance globale `backend_info` (type BackendInfo).

Usage :
    from core.backend import backend_info
    tensor = torch.zeros(10).to(backend_info.device)
    print(backend_info)           # résumé lisible
    print(backend_info.vram_mb)   # VRAM disponible (0 si CPU)
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# RÉSULTAT STRUCTURÉ
# ══════════════════════════════════════════════════════════════════════
@dataclass
class BackendInfo:
    """
    Décrit le backend GPU sélectionné et ses capacités.

    Attributes:
        name:          Identifiant du backend : "rocm" | "directml" | "cuda" | "cpu"
        device:        Objet torch.device prêt à l'emploi.
        device_name:   Nom lisible du périphérique (ex: "AMD Radeon RX 6700 XT").
        dtype_compute: dtype optimal pour l'inférence (torch.float16 / bfloat16 / float32).
        dtype_storage: dtype pour stocker les poids chargés (peut différer de dtype_compute).
        vram_mb:       VRAM totale détectée en mégaoctets (0 si CPU ou inconnu).
        vram_free_mb:  VRAM libre au moment de l'initialisation (0 si CPU).
        bench_ms:      Durée du micro-benchmark matmul en millisecondes.
        torch_version: Version de PyTorch détectée.
        extra:         Informations complémentaires spécifiques au backend.
    """

    name:          str
    device:        Any               # torch.device — typé Any pour éviter l'import au niveau module
    device_name:   str
    dtype_compute: Any               # torch.dtype
    dtype_storage: Any               # torch.dtype
    vram_mb:       int   = 0
    vram_free_mb:  int   = 0
    bench_ms:      float = 0.0
    torch_version: str   = ""
    extra:         dict  = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────────
    def quantization_level(self) -> str:
        """
        Retourne le niveau de quantization RWKV recommandé
        selon la VRAM disponible, en cohérence avec config.BACKEND.

        Returns:
            "none"  si VRAM ≥ 8 Go  (ou CPU — modèle chargé en float)
            "int8"  si 4 Go ≤ VRAM < 8 Go
            "int4"  si VRAM < 4 Go
        """
        from config import BACKEND
        if self.vram_mb == 0:          # CPU
            return "none"
        if self.vram_mb >= BACKEND.vram_medium_mb:
            return "none"
        if self.vram_mb >= BACKEND.vram_small_mb:
            return "int8"
        return "int4"

    # ──────────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        vram_str = (
            f"{self.vram_mb:,} Mo total / {self.vram_free_mb:,} Mo libre"
            if self.vram_mb else "N/A (CPU)"
        )
        return (
            f"Backend      : {self.name.upper()}\n"
            f"Périphérique : {self.device_name}\n"
            f"Device       : {self.device}\n"
            f"dtype compute: {self.dtype_compute}\n"
            f"VRAM         : {vram_str}\n"
            f"Benchmark    : {self.bench_ms:.1f} ms (matmul {_bench_size()}×{_bench_size()})\n"
            f"PyTorch      : {self.torch_version}\n"
            f"Quantization : {self.quantization_level()} (recommandée)"
        )


# ══════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ══════════════════════════════════════════════════════════════════════
def _bench_size() -> int:
    """Retourne la taille du matmul micro-benchmark depuis la config."""
    try:
        from config import BACKEND
        return BACKEND.benchmark_size
    except Exception:
        return 512


def _run_bench(torch_module: Any, device: Any, size: int) -> float:
    """
    Exécute un micro-benchmark matmul et retourne la durée en ms.
    Retourne float("inf") en cas d'erreur.

    Args:
        torch_module: Module torch importé.
        device:       Device cible (torch.device).
        size:         Dimension N du matmul N×N.

    Returns:
        Durée en millisecondes ou float("inf") si échec.
    """
    try:
        t = torch_module
        a = t.randn(size, size, device=device)
        b = t.randn(size, size, device=device)
        # Warm-up
        _ = t.mm(a, b)
        if hasattr(t, "cuda") and device.type == "cuda":
            t.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            _ = t.mm(a, b)
        if hasattr(t, "cuda") and device.type == "cuda":
            t.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / 3 * 1_000
        return round(elapsed_ms, 2)
    except Exception as exc:
        log.debug("Benchmark échoué sur %s : %s", device, exc)
        return float("inf")


def _vram_info(torch_module: Any, device: Any) -> tuple[int, int]:
    """
    Retourne (vram_total_mb, vram_free_mb) pour un device GPU.
    Retourne (0, 0) si l'information est indisponible.

    Args:
        torch_module: Module torch importé.
        device:       Device GPU (torch.device).

    Returns:
        Tuple (total_mb, free_mb).
    """
    try:
        t = torch_module
        if device.type == "cuda" and hasattr(t.cuda, "mem_get_info"):
            free_b, total_b = t.cuda.mem_get_info(device)
            return total_b // (1024 ** 2), free_b // (1024 ** 2)
        if device.type == "cuda" and hasattr(t.cuda, "get_device_properties"):
            props = t.cuda.get_device_properties(device)
            total_mb = props.total_memory // (1024 ** 2)
            return total_mb, total_mb   # free inconnu → approximation optimiste
    except Exception as exc:
        log.debug("Impossible de lire la VRAM : %s", exc)
    return 0, 0


def _optimal_dtype(torch_module: Any, device: Any) -> tuple[Any, Any]:
    """
    Détermine les dtypes compute et storage optimaux pour ce device.

    Priorité :
      - bfloat16 si supporté (AMD ROCm, NVIDIA Ampere+)
      - float16  sinon (GPU générique)
      - float32  en fallback CPU

    Args:
        torch_module: Module torch importé.
        device:       Device cible.

    Returns:
        Tuple (dtype_compute, dtype_storage).
    """
    t = torch_module
    if device.type == "cpu":
        return t.float32, t.float32

    try:
        from config import BACKEND
        if BACKEND.bf16_preferred and t.cuda.is_bf16_supported():
            return t.bfloat16, t.bfloat16
        if BACKEND.fp16_enabled:
            return t.float16, t.float16
    except Exception:
        pass

    # Fallback : tenter bfloat16 manuellement
    try:
        _ = t.zeros(1, dtype=t.bfloat16, device=device)
        return t.bfloat16, t.bfloat16
    except Exception:
        pass

    try:
        _ = t.zeros(1, dtype=t.float16, device=device)
        return t.float16, t.float16
    except Exception:
        pass

    return t.float32, t.float32


# ══════════════════════════════════════════════════════════════════════
# SONDES PAR BACKEND (retournent BackendInfo ou None si indisponible)
# ══════════════════════════════════════════════════════════════════════
def _probe_rocm() -> BackendInfo | None:
    """
    Tente de détecter un GPU AMD via PyTorch + ROCm (HIP).
    Sur ROCm, le device exposé est "cuda" (HIP émule l'API CUDA).

    Returns:
        BackendInfo configuré pour ROCm, ou None si non disponible.
    """
    if platform.system() != "Linux":
        log.debug("ROCm probe ignoré (non-Linux).")
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        # Vérifier que c'est bien ROCm et non CUDA
        device_name = torch.cuda.get_device_name(0)
        # ROCm expose des noms contenant "gfx" ou "Radeon" ou "AMD"
        # Note : on vérifie aussi la variable d'environnement HIP
        is_rocm = (
            "AMD" in device_name.upper()
            or "RADEON" in device_name.upper()
            or "GFX" in device_name.upper()
            or hasattr(torch.version, "hip")
            or torch.version.hip is not None  # type: ignore[union-attr]
        )
        if not is_rocm:
            log.debug("CUDA disponible mais pas ROCm — probe ROCm ignoré.")
            return None

        device        = torch.device("cuda:0")
        size          = _bench_size()
        bench_ms      = _run_bench(torch, device, size)
        vram, vfree   = _vram_info(torch, device)
        dt_c, dt_s    = _optimal_dtype(torch, device)

        log.info("ROCm détecté : %s (VRAM %d Mo, bench %.1f ms)", device_name, vram, bench_ms)
        return BackendInfo(
            name          = "rocm",
            device        = device,
            device_name   = device_name,
            dtype_compute = dt_c,
            dtype_storage = dt_s,
            vram_mb       = vram,
            vram_free_mb  = vfree,
            bench_ms      = bench_ms,
            torch_version = torch.__version__,
            extra         = {"hip_version": getattr(torch.version, "hip", "unknown")},
        )
    except Exception as exc:
        log.debug("Probe ROCm échoué : %s", exc)
        return None


def _probe_directml() -> BackendInfo | None:
    """
    Tente de détecter un GPU AMD (ou tout GPU compatible) via torch-directml
    (Windows, via DirectX 12 / DirectML).

    Returns:
        BackendInfo configuré pour DirectML, ou None si non disponible.
    """
    if platform.system() != "Windows":
        log.debug("DirectML probe ignoré (non-Windows).")
        return None
    try:
        import torch
        import torch_directml  # type: ignore[import]

        device      = torch_directml.device()
        device_name = torch_directml.device_name(0)
        size        = _bench_size()
        bench_ms    = _run_bench(torch, device, size)
        # DirectML n'expose pas d'API VRAM standard — on tente WMI en fallback
        vram, vfree = _vram_directml_vram()
        dt_c        = torch.float16   # DirectML supporte fp16, pas bf16 en général
        dt_s        = torch.float16

        log.info("DirectML détecté : %s (VRAM %d Mo, bench %.1f ms)", device_name, vram, bench_ms)
        return BackendInfo(
            name          = "directml",
            device        = device,
            device_name   = device_name,
            dtype_compute = dt_c,
            dtype_storage = dt_s,
            vram_mb       = vram,
            vram_free_mb  = vfree,
            bench_ms      = bench_ms,
            torch_version = torch.__version__,
            extra         = {"directml_version": getattr(torch_directml, "__version__", "unknown")},
        )
    except Exception as exc:
        log.debug("Probe DirectML échoué : %s", exc)
        return None


def _vram_directml_vram() -> tuple[int, int]:
    """
    Tente de lire la VRAM via WMI (Windows Management Instrumentation).
    Retourne (0, 0) si l'information n'est pas accessible.

    Returns:
        Tuple (total_mb, free_mb) — free_mb toujours 0 via WMI.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-WmiObject Win32_VideoController).AdapterRAM"],
            capture_output=True, text=True, timeout=5
        )
        raw = result.stdout.strip()
        if raw and raw.isdigit():
            total_mb = int(raw) // (1024 ** 2)
            return total_mb, 0
    except Exception:
        pass
    return 0, 0


def _probe_cuda() -> BackendInfo | None:
    """
    Détecte un GPU NVIDIA via PyTorch CUDA standard.
    Ce probe est ignoré si ROCm a déjà été sélectionné (même device "cuda").

    Returns:
        BackendInfo configuré pour CUDA, ou None si non disponible.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        device_name = torch.cuda.get_device_name(0)
        # Si c'est en fait un GPU AMD (détecté via ROCm), on ne reprend pas ici
        is_amd = (
            "AMD" in device_name.upper()
            or "RADEON" in device_name.upper()
            or getattr(torch.version, "hip", None) is not None
        )
        if is_amd:
            log.debug("GPU AMD détecté depuis probe CUDA — sera géré par probe ROCm.")
            return None

        device       = torch.device("cuda:0")
        size         = _bench_size()
        bench_ms     = _run_bench(torch, device, size)
        vram, vfree  = _vram_info(torch, device)
        dt_c, dt_s   = _optimal_dtype(torch, device)

        log.info("CUDA détecté : %s (VRAM %d Mo, bench %.1f ms)", device_name, vram, bench_ms)
        return BackendInfo(
            name          = "cuda",
            device        = device,
            device_name   = device_name,
            dtype_compute = dt_c,
            dtype_storage = dt_s,
            vram_mb       = vram,
            vram_free_mb  = vfree,
            bench_ms      = bench_ms,
            torch_version = torch.__version__,
            extra         = {"cuda_version": torch.version.cuda or "unknown"},
        )
    except Exception as exc:
        log.debug("Probe CUDA échoué : %s", exc)
        return None


def _probe_cpu() -> BackendInfo:
    """
    Fallback universel : crée un BackendInfo pour le CPU.
    Ne peut pas échouer.

    Returns:
        BackendInfo configuré pour le CPU.
    """
    try:
        import torch
        device    = torch.device("cpu")
        size      = _bench_size()
        bench_ms  = _run_bench(torch, device, size)
        version   = torch.__version__
        dtype     = torch.float32
    except Exception:
        # torch non installé du tout (très rare)
        device, bench_ms, version, dtype = "cpu", 0.0, "N/A", None

    cpu_name = platform.processor() or platform.machine() or "CPU inconnu"
    log.info("Backend CPU sélectionné : %s", cpu_name)
    return BackendInfo(
        name          = "cpu",
        device        = device,
        device_name   = cpu_name,
        dtype_compute = dtype,
        dtype_storage = dtype,
        vram_mb       = 0,
        vram_free_mb  = 0,
        bench_ms      = bench_ms,
        torch_version = version,
        extra         = {"platform": platform.platform()},
    )


# ══════════════════════════════════════════════════════════════════════
# SÉLECTION AUTOMATIQUE DU BACKEND
# ══════════════════════════════════════════════════════════════════════
def detect_backend() -> BackendInfo:
    """
    Détecte et initialise le meilleur backend GPU disponible.

    Ordre de priorité (configurable via ZYLOS_BACKEND) :
      1. ROCm      — AMD Linux (torch + HIP)
      2. DirectML  — AMD Windows (torch-directml)
      3. CUDA      — NVIDIA
      4. CPU       — fallback

    Si BACKEND.forced != "auto", le backend demandé est forcé directement
    et les autres sondes sont ignorées. Une erreur de configuration provoque
    un fallback sur CPU avec un avertissement.

    Returns:
        BackendInfo décrivant le backend sélectionné.

    Raises:
        Ne lève jamais d'exception — le CPU est le filet de sécurité final.
    """
    from config import BACKEND

    forced = BACKEND.forced.lower().strip()

    # ── Backend forcé via variable d'environnement ────────────────────
    if forced != "auto":
        log.info("Backend forcé : ZYLOS_BACKEND=%s", forced)
        probe_map = {
            "rocm":      _probe_rocm,
            "directml":  _probe_directml,
            "cuda":      _probe_cuda,
            "cpu":       _probe_cpu,
        }
        prober = probe_map.get(forced)
        if prober is None:
            log.warning(
                "Valeur ZYLOS_BACKEND inconnue : '%s'. "
                "Valeurs valides : rocm, directml, cuda, cpu. "
                "Fallback → détection automatique.",
                forced
            )
        else:
            result = prober()
            if result is not None:
                return result
            log.warning(
                "Backend forcé '%s' indisponible sur cette machine. "
                "Fallback → détection automatique.",
                forced
            )

    # ── Détection automatique ─────────────────────────────────────────
    log.info("Détection automatique du backend GPU…")

    # Toutes les sondes dans l'ordre de priorité
    probes = [_probe_rocm, _probe_directml, _probe_cuda]
    candidates: list[BackendInfo] = []

    for probe in probes:
        try:
            result = probe()
            if result is not None:
                candidates.append(result)
        except Exception as exc:
            log.debug("Sonde %s exception inattendue : %s", probe.__name__, exc)

    if not candidates:
        log.info("Aucun GPU détecté — utilisation du CPU.")
        return _probe_cpu()

    if len(candidates) == 1:
        log.info("Un seul backend GPU disponible : %s", candidates[0].name.upper())
        return candidates[0]

    # Plusieurs backends détectés → choisir le plus rapide au benchmark
    best = min(candidates, key=lambda b: b.bench_ms)
    log.info(
        "Plusieurs backends disponibles (%s) — sélection du plus rapide : %s (%.1f ms)",
        ", ".join(c.name for c in candidates),
        best.name.upper(),
        best.bench_ms,
    )
    return best


# ══════════════════════════════════════════════════════════════════════
# INITIALISATION SINGLETON
# ══════════════════════════════════════════════════════════════════════
def _initialize() -> BackendInfo:
    """
    Point d'entrée interne : détecte le backend, log le résultat,
    et pousse les métriques matériel dans le registre global.

    Returns:
        BackendInfo prêt à l'emploi.
    """
    log.debug("Initialisation de core/backend.py…")
    info = detect_backend()

    log.info(
        "Backend sélectionné : %s | %s | VRAM %d Mo | bench %.1f ms | quant %s",
        info.name.upper(),
        info.device_name,
        info.vram_mb,
        info.bench_ms,
        info.quantization_level(),
    )

    # Persistance dans le registre de métriques (section "_modules/backend")
    try:
        metrics.update_module("backend", {
            "name":         info.name,
            "device_name":  info.device_name,
            "device":       str(info.device),
            "vram_total_mb": info.vram_mb,
            "vram_free_mb": info.vram_free_mb,
            "bench_ms":     info.bench_ms,
            "dtype_compute": str(info.dtype_compute),
            "quantization": info.quantization_level(),
            "torch_version": info.torch_version,
        }, flush=False)   # flush différé : metrics.init() n'a peut-être pas encore eu lieu
    except Exception as exc:
        log.debug("Impossible de pousser les métriques backend (normal au 1er démarrage) : %s", exc)

    return info


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE — importée par tous les modules
# ══════════════════════════════════════════════════════════════════════
#: Singleton BackendInfo initialisé au premier import de ce module.
#: Usage : from core.backend import backend_info
backend_info: BackendInfo = _initialize()


# ══════════════════════════════════════════════════════════════════════
# API PUBLIQUE UTILITAIRE
# ══════════════════════════════════════════════════════════════════════
def get_device() -> Any:
    """
    Retourne le torch.device du backend actif.

    Raccourci pour : backend_info.device

    Returns:
        torch.device prêt à l'emploi.

    Example:
        >>> device = get_device()
        >>> tensor = torch.zeros(10).to(device)
    """
    return backend_info.device


def get_dtype() -> Any:
    """
    Retourne le dtype de calcul optimal (torch.dtype).

    Raccourci pour : backend_info.dtype_compute

    Returns:
        torch.dtype (bfloat16, float16 ou float32).

    Example:
        >>> model = MyModel().to(dtype=get_dtype())
    """
    return backend_info.dtype_compute


def is_gpu_available() -> bool:
    """
    Retourne True si le backend actif utilise un GPU (pas CPU).

    Returns:
        True si backend = rocm | directml | cuda.

    Example:
        >>> if is_gpu_available():
        ...     log.info("Accélération GPU active")
    """
    return backend_info.name != "cpu"


def to_device(tensor: Any) -> Any:
    """
    Déplace un tenseur vers le device du backend actif.

    Équivalent à tensor.to(backend_info.device), mais utilisable
    sans importer torch explicitement dans les modules appelants.

    Args:
        tensor: Tenseur torch à déplacer.

    Returns:
        Tenseur sur le bon device.

    Example:
        >>> from core.backend import to_device
        >>> x = to_device(torch.randn(4, 512))
    """
    return tensor.to(backend_info.device)


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python core/backend.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — core/backend.py  smoke test")
    print("═" * 60)
    print()
    print(backend_info)
    print()

    # Vérification des types de retour
    assert backend_info.name in ("rocm", "directml", "cuda", "cpu"), \
        f"Nom backend invalide : {backend_info.name!r}"
    assert backend_info.device is not None, "device est None"
    assert backend_info.dtype_compute is not None, "dtype_compute est None"
    assert isinstance(backend_info.bench_ms, float), "bench_ms doit être float"
    assert isinstance(backend_info.vram_mb, int), "vram_mb doit être int"
    assert backend_info.quantization_level() in ("none", "int8", "int4"), \
        f"Niveau de quantization invalide : {backend_info.quantization_level()!r}"

    # API utilitaire
    device = get_device()
    dtype  = get_dtype()
    gpu    = is_gpu_available()

    assert device is not None, "get_device() a retourné None"
    assert dtype  is not None, "get_dtype() a retourné None"
    assert isinstance(gpu, bool), "is_gpu_available() doit retourner un bool"

    # Test to_device si torch est disponible
    try:
        import torch
        t = torch.randn(4, 4)
        t2 = to_device(t)
        assert str(t2.device).startswith(str(backend_info.device).split(":")[0]), \
            f"to_device() device inattendu : {t2.device}"
        print(f"✅  to_device() : tenseur sur {t2.device}")
    except ImportError:
        print("⚠   torch non disponible — test to_device ignoré.")

    print("\n✅  Tous les tests core/backend.py sont passés.")
    print(f"   Backend actif     : {backend_info.name.upper()}")
    print(f"   Quantization RWKV : {backend_info.quantization_level()}")
    print(f"   GPU disponible    : {is_gpu_available()}")