# ZYLOS AI

**IA locale non-Transformer auto-apprenante — RWKV-7 + QLoRA + RAG**

> Fonctionne 100% hors-ligne. Aucune donnée ne quitte ta machine (sauf appels Mistral optionnels pour l'auto-amélioration du code).

---

## Vue d'ensemble

ZYLOS AI est un assistant conversationnel local qui :

- répond en français via **RWKV-7 World** (modèle récurrent, non-Transformer)
- enrichit ses réponses par **RAG** sur une mémoire vectorielle ChromaDB
- **raisonne étape par étape** grâce au chain-of-thought intégré
- **apprend chaque nuit** en scrappant le web et fine-tunant via QLoRA
- **s'auto-améliore** en analysant son propre code source via Mistral Codestral
- tourne sur **GPU AMD** (DirectML Windows / ROCm Linux), NVIDIA ou CPU

---

## Architecture

```
main.py                  ← point d'entrée (chat + scheduler)
├── core/
│   ├── backend.py       ← détection GPU (ROCm / DirectML / CUDA / CPU)
│   ├── model.py         ← chargement RWKV-7, génération, streaming, embeddings
│   └── tokenizer.py     ← WorldTokenizer (65 536 tokens, Unicode complet)
├── modules/
│   ├── brain.py         ← orchestration RAG + historique + chain-of-thought
│   ├── vectordb.py      ← ChromaDB local, embeddings RWKV, déduplication
│   ├── scraper.py       ← web scraper (trafilatura + BS4), chunking intelligent
│   ├── trainer.py       ← fine-tuning QLoRA, replay buffer, rollback automatique
│   ├── improver.py      ← auto-amélioration code via Mistral Codestral
│   └── scheduler.py     ← pipeline journalier automatique (03:00 UTC)
├── pipeline/
│   ├── corpus_builder.py ← JSONL d'entraînement depuis pages scrapées
│   ├── replay_buffer.py  ← tampon circulaire anti-oubli catastrophique
│   └── daily_run.py      ← exécution manuelle du pipeline
├── api/
│   └── mistral_client.py ← client Mistral (USAGE EXCLUSIF : improver.py)
├── utils/
│   ├── logger.py         ← logs colorés + rotation quotidienne
│   ├── metrics.py        ← registre centralisé, persistance JSON
│   └── backup.py         ← snapshots ZIP horodatés
└── config.py             ← source unique de vérité pour tous les paramètres
```

---

## Prérequis

- **Python 3.10+**
- **GPU AMD recommandé** (4 Go VRAM minimum pour le modèle 1.5B)
  - Windows : `torch-directml`
  - Linux : PyTorch + ROCm
- **GPU NVIDIA** : PyTorch CUDA standard
- **CPU** : fonctionne, mais lent (~2–5 tok/s)

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/ton-repo/zylos-ai.git
cd zylos-ai

# 2. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Windows AMD uniquement) Installer torch-directml
pip install torch==2.1.0 torch-directml

# 5. (Optionnel) Clé API Mistral pour l'auto-amélioration du code
export MISTRAL_API_KEY="sk-..."   # Linux/macOS
set MISTRAL_API_KEY=sk-...        # Windows CMD
```

Le modèle RWKV-7 (~3 Go pour 1.5B) est **téléchargé automatiquement** via `huggingface_hub` au premier lancement.

> ⚠️ **Si le téléchargement échoue**, récupère manuellement le fichier :  
> `https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth`  
> et place-le dans `data/models/`.

---

## Utilisation

```bash
# Chat interactif (mode normal)
python main.py

# Sans scheduler automatique
python main.py --no-sched

# Afficher les métriques
python main.py --stats

# Exécuter le pipeline d'apprentissage une fois
python main.py --once

# Pipeline manuel avec étapes choisies
python pipeline/daily_run.py --steps scraping,indexing,corpus_build

# Simulation sans exécution (debug)
python pipeline/daily_run.py --dry-run
```

### Commandes disponibles dans le chat

| Commande | Action |
|---|---|
| `/help` | Afficher l'aide |
| `/reset` | Effacer l'historique de conversation |
| `/stats` | Afficher les métriques système |
| `/quit` | Quitter |

---

## Configuration

Tous les paramètres sont dans `config.py`. Les variables d'environnement surchargent les valeurs par défaut.

| Variable | Défaut | Description |
|---|---|---|
| `ZYLOS_BACKEND` | `auto` | Forcer le backend : `rocm`, `directml`, `cuda`, `cpu` |
| `ZYLOS_MODEL_SIZE` | `1.5B` | Taille du modèle : `0.4B`, `1.5B`, `3B` |
| `ZYLOS_MODEL_FILE` | *(auto)* | Chemin vers un fichier `.pth` local |
| `ZYLOS_LOG_LEVEL` | `INFO` | Niveau de log : `DEBUG`, `INFO`, `WARNING` |
| `ZYLOS_SYSTEM_PROMPT` | *(voir config)* | Prompt système personnalisé |
| `ZYLOS_COT` | `true` | Active le chain-of-thought (raisonnement étape par étape) |
| `ZYLOS_AUTO_APPLY` | `false` | Appliquer les suggestions de code automatiquement (**danger**) |
| `MISTRAL_API_KEY` | *(vide)* | Clé API Mistral (optionnel, pour l'auto-amélioration) |

### Choix de la taille de modèle selon la VRAM

| VRAM disponible | Modèle recommandé | Quantization |
|---|---|---|
| < 4 Go | 0.4B | int4 |
| 4 – 8 Go | 1.5B | int8 |
| > 8 Go | 3B | none (fp16) |

---

## Améliorations du raisonnement (v2)

### Chain-of-thought
ZYLOS peut raisonner étape par étape avant de répondre. Activé automatiquement sur les questions complexes (code, maths, explications, "pourquoi", "comment").

Le modèle produit un bloc `<think>...</think>` interne (non affiché à l'utilisateur) puis formule sa réponse. Ce mécanisme améliore significativement la qualité des réponses analytiques.

**Désactiver** si tu veux maximiser la vitesse :
```bash
set ZYLOS_COT=false   # Windows
export ZYLOS_COT=false # Linux/macOS
```

### Paramètres d'inférence optimisés
| Paramètre | Ancien | Nouveau | Effet |
|---|---|---|---|
| `temperature` | 0.8 | **0.7** | Réponses plus précises |
| `repeat_penalty` | 1.1 | **1.2** | Moins de répétitions |
| `top_p` | 0.9 | **0.85** | Meilleure cohérence |
| `max_tokens` | 512 | **768** | Réponses plus complètes |
| `max_history_turns` | 10 | **12** | Meilleur contexte conversationnel |

### Prompt système enrichi
Le prompt système guide maintenant le modèle vers un raisonnement structuré : décomposer les problèmes, signaler l'incertitude, citer les sources.

---

## Pipeline d'apprentissage automatique

Le scheduler se déclenche chaque nuit à **03:00 UTC** et exécute 7 étapes :

```
1. scraping       → collecte des sources web configurées
2. indexing       → ajout des chunks dans ChromaDB
3. corpus_build   → construction des exemples JSONL
4. training       → fine-tuning QLoRA (200 steps, batch 4×4)
5. evaluation     → test rapide post-training
6. improvement    → analyse code via Mistral Codestral (si clé configurée)
7. report         → sauvegarde des métriques dans data/logs/
```

---

## Anti-oubli catastrophique

Le fine-tuning continu réutilise **30% d'anciens exemples** (replay buffer, 1 000 échantillons max) mélangés aux nouveaux. Si la loss finale dépasse la baseline de plus de 5%, un **rollback automatique** restaure les poids LoRA précédents.

---

## Auto-amélioration du code

> ⚠️ Ce mécanisme modifie le code source du projet. Comprendre son fonctionnement avant d'activer `ZYLOS_AUTO_APPLY`.

Toutes les N interactions (configurable), `improver.py` envoie un module Python à Mistral Codestral avec les métriques associées. Chaque suggestion est :

1. Validée syntaxiquement (`ast.parse` + `py_compile`)
2. Sauvegardée dans `data/backups/pending/`
3. Appliquée seulement si `ZYLOS_AUTO_APPLY=true` (sinon, validation manuelle)

---

## Limitations actuelles

- Le modèle RWKV-7 **1.5B** ne rivalise pas encore avec GPT-4 sur le raisonnement complexe.  
  Le chain-of-thought améliore cela, mais la taille du modèle reste un facteur limitant.
- Pour se rapprocher de GPT-4, utilise le **modèle 3B** si ta VRAM le permet (>8 Go).
- Le fine-tuning QLoRA nécessite le package `peft` (HuggingFace) ; sans lui, une boucle simplifiée est utilisée.
- DirectML ne supporte pas bfloat16 — float16 est utilisé sur Windows AMD.
- La progression vers GPT-4 nécessitera l'accumulation de corpus, de sessions d'entraînement et potentiellement un modèle de base plus grand.

---

## Corrections v2

- **Fix critique** : URL de téléchargement HuggingFace corrigée (`/resolve/main/` au lieu de `/tree/main/`)
- **Nouveau** : `huggingface_hub` pour des téléchargements fiables avec reprise automatique
- **Nouveau** : Chain-of-thought pour un meilleur raisonnement
- **Nouveau** : Paramètres d'inférence optimisés
- **Nouveau** : Prompt système enrichi

---

## Licence

Usage personnel / éducatif. Voir LICENSE pour les détails.