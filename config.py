"""
config.py — Configuration centralisée de ZYLOS AI
====================================================
Source unique de vérité pour tous les paramètres du projet.
Aucune valeur de configuration ne doit être codée en dur ailleurs.

RÈGLE ABSOLUE : Mistral API est réservée EXCLUSIVEMENT à modules/improver.py
                (analyse et modification du code source du projet).
                Toutes les réponses utilisateur passent par RWKV local.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ─────────────────────────────────────────────
# Racine du projet (dossier contenant config.py)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"


# ══════════════════════════════════════════════════════════════════════
# 1. CHEMINS
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PathsConfig:
    root:         Path = PROJECT_ROOT
    data:         Path = DATA_DIR
    chroma_db:    Path = DATA_DIR / "chroma_db"
    models:       Path = DATA_DIR / "models"
    corpus:       Path = DATA_DIR / "corpus"
    lora_weights: Path = DATA_DIR / "lora_weights"
    backups:      Path = DATA_DIR / "backups"
    logs:         Path = DATA_DIR / "logs"
    metrics_file: Path = DATA_DIR / "metrics.json"

    def create_all(self) -> None:
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, Path) and val.suffix == "":
                val.mkdir(parents=True, exist_ok=True)


PATHS = PathsConfig()


# ══════════════════════════════════════════════════════════════════════
# 2. GPU / BACKEND
# ══════════════════════════════════════════════════════════════════════
BackendType = Literal["rocm", "directml", "cuda", "cpu"]

@dataclass(frozen=True)
class BackendConfig:
    forced:         str  = field(default_factory=lambda: os.getenv("ZYLOS_BACKEND", "auto"))
    benchmark_size: int  = 512
    fp16_enabled:   bool = True
    bf16_preferred: bool = True
    vram_small_mb:  int  = 4_000
    vram_medium_mb: int  = 8_000


BACKEND = BackendConfig()


# ══════════════════════════════════════════════════════════════════════
# 3. RWKV — moteur local principal
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RWKVConfig:
    hf_repo:           str          = "BlinkDL/rwkv-7-world"
    default_size:      str          = field(default_factory=lambda: os.getenv("ZYLOS_MODEL_SIZE", "1.5B"))
    local_model_file:  str | None   = field(default_factory=lambda: os.getenv("ZYLOS_MODEL_FILE", None))

    # ─── Génération ──────────────────────────────────────────────────
    # Contexte étendu pour un meilleur raisonnement multi-tours
    context_length:  int   = 4096

    # Température plus basse = réponses plus précises et cohérentes
    # 0.7 est un bon équilibre créativité/précision pour un assistant
    temperature:     float = 0.7

    # top_p plus strict = meilleure cohérence des réponses longues
    top_p:           float = 0.85

    # top_k élargi = plus de diversité de vocabulaire
    top_k:           int   = 50

    # Repeat penalty plus fort = évite les répétitions fastidieuses
    repeat_penalty:  float = 1.2

    # Tokens max augmentés pour des réponses plus complètes
    max_tokens:      int   = 768

    quantization:    str   = "auto"
    persist_state:   bool  = True
    state_file:      Path  = DATA_DIR / "rwkv_state.pt"
    streaming:       bool  = True
    embedding_layer: str   = "last_hidden"


RWKV = RWKVConfig()


# ══════════════════════════════════════════════════════════════════════
# 4. MISTRAL API — USAGE RESTREINT : improver.py UNIQUEMENT
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MistralConfig:
    api_key:              str   = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    base_url:             str   = "https://api.mistral.ai/v1"
    code_model:           str   = "codestral-latest"
    requests_per_sec:     float = 1.0
    max_retries:          int   = 4
    retry_base_delay:     float = 1.0
    monthly_token_budget: int   = 1_000_000_000
    alert_threshold_pct:  float = 0.80
    max_code_chars:       int   = 6_000

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


MISTRAL = MistralConfig()


# ══════════════════════════════════════════════════════════════════════
# 5. SCRAPING
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ScraperConfig:
    delay_between_requests: float = 2.0
    max_pages_per_session:  int   = 50
    max_tokens_per_page:    int   = 2_000
    request_timeout_sec:    int   = 15
    respect_robots_txt:     bool  = True
    user_agent:             str   = (
        "ZylosAI/1.0 (+https://github.com/zylos-ai; educational crawler)"
    )
    default_sources: tuple[str, ...] = (
        "https://fr.wikipedia.org/wiki/Portail:Informatique",
        "https://arxiv.org/list/cs.AI/recent",
        "https://news.ycombinator.com",
    )
    accepted_languages: tuple[str, ...] = ("fr", "en")


SCRAPER = ScraperConfig()


# ══════════════════════════════════════════════════════════════════════
# 6. MÉMOIRE VECTORIELLE
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class VectorDBConfig:
    collection_name:            str   = "zylos_memory"
    chunk_size_tokens:          int   = 500
    chunk_overlap_tokens:       int   = 50
    top_k_results:              int   = 5
    dedup_similarity_threshold: float = 0.98
    embedding_backend:          str   = "rwkv"


VECTORDB = VectorDBConfig()


# ══════════════════════════════════════════════════════════════════════
# 7. CERVEAU — raisonnement amélioré
# ══════════════════════════════════════════════════════════════════════

# Prompt système renforcé pour un meilleur raisonnement
# Inspiré des techniques de prompting chain-of-thought
_DEFAULT_SYSTEM_PROMPT = (
    "Tu es Zylos, une IA locale non-Transformer auto-apprenante, "
    "développée sur l'architecture RWKV-7. "
    "Tu réponds en français par défaut, de façon précise et structurée. "
    "\n\n"
    "Principes de raisonnement :\n"
    "- Réfléchis étape par étape avant de répondre aux questions complexes\n"
    "- Si tu n'es pas sûr, dis-le clairement plutôt que d'inventer\n"
    "- Cite tes sources mémoire entre crochets : [Source: <titre>]\n"
    "- Pour les problèmes de code ou de logique, décompose le problème\n"
    "- Sois concis mais complet : préfère la qualité à la quantité\n"
    "\n"
    "Capacités : raisonnement, code, analyse, explications, créativité."
)

@dataclass(frozen=True)
class BrainConfig:
    max_history_turns:  int   = 12   # +2 tours pour un meilleur contexte
    max_history_tokens: int   = 4_000
    rag_top_k:          int   = 5
    rag_min_score:      float = 0.55  # légèrement abaissé pour plus de contexte RAG

    # Chain-of-thought : ajoute "<think>" dans le prompt pour guider le raisonnement
    chain_of_thought:   bool  = field(
        default_factory=lambda: os.getenv("ZYLOS_COT", "true").lower() == "true"
    )

    system_prompt: str = field(default_factory=lambda: os.getenv(
        "ZYLOS_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT
    ))


BRAIN = BrainConfig()


# ══════════════════════════════════════════════════════════════════════
# 8. ENTRAÎNEMENT QLoRA
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TrainerConfig:
    lora_rank:     int   = 16
    lora_alpha:    int   = 32
    lora_dropout:  float = 0.05
    target_modules: tuple[str, ...] = (
        "time_mix_key", "time_mix_value",
        "time_mix_receptance", "time_mix_gate"
    )
    learning_rate:          float = 2e-4
    batch_size:             int   = 4
    gradient_accum:         int   = 4
    max_steps:              int   = 200
    warmup_steps:           int   = 20
    gradient_checkpointing: bool  = True
    replay_buffer_size:     int   = 1_000
    replay_ratio:           float = 0.30
    rollback_on_regression: bool  = True
    regression_threshold:   float = 0.05


TRAINER = TrainerConfig()


# ══════════════════════════════════════════════════════════════════════
# 9. AUTO-AMÉLIORATION DU CODE
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ImproverConfig:
    enabled:             bool  = True
    trigger_every_n:     int   = 50
    auto_apply:          bool  = field(
        default_factory=lambda: os.getenv("ZYLOS_AUTO_APPLY", "false").lower() == "true"
    )
    pending_dir:         Path  = DATA_DIR / "backups" / "pending"
    excluded_modules:    tuple[str, ...] = ("config.py", "utils/backup.py")
    min_confidence_score: float = 0.70


IMPROVER = ImproverConfig()


# ══════════════════════════════════════════════════════════════════════
# 10. SCHEDULER
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SchedulerConfig:
    daily_run_hour:   int = 3
    daily_run_minute: int = 0
    step_timeouts: dict[str, int] = field(default_factory=lambda: {
        "scraping":     30 * 60,
        "indexing":     10 * 60,
        "corpus_build": 20 * 60,
        "training":     90 * 60,
        "evaluation":    5 * 60,
        "improvement":   5 * 60,
        "report":        2 * 60,
    })
    max_retries:     int = 3
    retry_delay_sec: int = 30


SCHEDULER = SchedulerConfig()


# ══════════════════════════════════════════════════════════════════════
# 11. LOGGING
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LogConfig:
    level:          str  = field(default_factory=lambda: os.getenv("ZYLOS_LOG_LEVEL", "INFO").upper())
    format:         str  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    date_format:    str  = "%Y-%m-%d %H:%M:%S"
    max_log_days:   int  = 30
    console_output: bool = True


LOG = LogConfig()


# ══════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════
def validate_config() -> list[str]:
    warnings: list[str] = []
    if not MISTRAL.is_configured():
        warnings.append("MISTRAL_API_KEY non définie — modules/improver.py sera désactivé.")
    if IMPROVER.auto_apply:
        warnings.append(
            "⚠️  ZYLOS_AUTO_APPLY=true : les modifications de code source seront "
            "appliquées AUTOMATIQUEMENT sans validation humaine."
        )
    if RWKV.quantization == "auto":
        warnings.append(
            "Quantization RWKV en mode 'auto' — déterminée au runtime "
            "selon la VRAM disponible."
        )
    return warnings


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE DE TEST RAPIDE
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 62)
    print("  ZYLOS AI — Validation de la configuration")
    print("═" * 62)

    issues = validate_config()
    if issues:
        print("\n⚠  Avertissements :")
        for w in issues:
            print(f"   • {w}")
    else:
        print("\n✅  Configuration valide, aucun avertissement.")

    print(f"\n📁  Racine projet    : {PATHS.root}")
    print(f"📂  Données          : {PATHS.data}")
    print(f"🤖  Modèle RWKV      : {RWKV.default_size} — {RWKV.hf_repo}")
    print(f"🌡️  Température       : {RWKV.temperature} (raisonnement optimisé)")
    print(f"🔄  Repeat penalty   : {RWKV.repeat_penalty}")
    print(f"🧠  Chain-of-thought  : {'✅ activé' if BRAIN.chain_of_thought else '❌ désactivé'}")
    print(f"🧠  Backend GPU      : {BACKEND.forced}")
    print(f"🔑  Mistral API      : {'✅ configurée' if MISTRAL.is_configured() else '❌ absente (improver désactivé)'}")
    print(f"🛠   Auto-apply code  : {'⚠  ACTIVÉ' if IMPROVER.auto_apply else '✅ désactivé (safe)'}")
    print(f"⏰  Run journalier   : {SCHEDULER.daily_run_hour:02d}:{SCHEDULER.daily_run_minute:02d}")
    print()