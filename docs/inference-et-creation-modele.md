# Inférence & Création de modèle — fiche technique

Date : 07/10/2026 — Projet RAG BTS CIEL.

## 1. Inférence CPU en 2-bit / ternaire {-1, 0, +1}

**Correction : ce n'est pas NVIDIA, c'est Microsoft — BitNet / bitnet.cpp**
(`github.com/microsoft/bitnet`, licence MIT).

- Chaque poids vaut -1, 0 ou +1 → log2(3) ≈ **1.58 bit/poids** (stocké sur 2 bits).
- Kernels CPU optimisés x86/ARM (lookup tables, pas de float dans le hot path) :
  2-6× plus rapide que le full-précision sur CPU, un 100B tourne à ~5-7 tok/s sur un simple CPU.
- Modèles natifs : `BitNet-b1.58-2B-4T` (2B params, ~0.4 GB, entraîné sur 4000 Mds de tokens).
  Aussi : modèles embedding 1-bit (`BitNet-embedding-0.6B/270M`) — piste pour notre RAG.
- **Limite clé : on ne convertit pas n'importe quel modèle en {-1,0,+1}.**
  Les modèles ternaires doivent être *entraînés* ternaires. Quantifier un modèle
  existant en ternaire post-hoc = qualité effondrée.

Tester sur le Mac mini :

```bash
git clone --recursive https://github.com/microsoft/bitnet.git && cd bitnet
pip install -r requirements.txt && python setup_env.py -md BitNet-b1.58-2B-4T -quant i2_s
python run_inference.py -m BitNet-b1.58-2B-4T -p "Explique le registre ADMUX" -t 4
```

## 2. Créer notre modèle : distillation parent → enfant

Avec ~50-80 MB de corpus, **pas de modèle from scratch** (il faut des trillions de tokens).
La voie réaliste : **teacher-student + LoRA**.

```
Corpus (chunks RAG)
   │  parent = gros modèle déjà sur le mini (gemma4:12b, gpt-oss:20b)
   ▼  génère des paires question/réponse (self-instruct)
Dataset Q/R (JSONL, ex. 5-10k paires)
   │  enfant = petit modèle (Qwen3-4B, gemma3:4b) + fine-tune LoRA via MLX (idéal M4)
   ▼
Modèle fusionné → quantif GGUF (Q4_K_M, voire Q2_K) → Ollama → rag-web / rag_test.py
```

Commandes (sur le Mac mini) :

```bash
# 1. Fine-tune LoRA
pip install mlx-lm
python -m mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \
  --train --data ./dataset_qa --iters 1000 --batch-size 4 --adapter-path ./adapters

# 2. Fusion + import Ollama
python -m mlx_lm.fuse --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters --save-path ./ciel-4b
ollama create ciel-4b -f Modelfile

# 3. Évaluation avec notre suite existante
python3 rag_test.py --models ciel-4b --output rapport_ciel-4b.pdf
```

## 3. À quoi s'attendre (honnêtement)

- Le fine-tune apporte surtout le **format** : citations, français, refus hors-corpus,
  style BTS. Les **connaissances** viennent déjà du RAG.
- Pour un usage RAG, l'ordre ROI : corpus complet → dataset Q/R teacher →
  LoRA → comparaison `rag_test.py` vs `gemma3:4b` baseline.
- Le ternaire BitNet est une cible d'**inférence** (déploiement léger), pas une méthode
  d'entraînement sur petit corpus.

## 4. Roadmap

1. Finir le corpus (`rag-data/` → `./rag index` + `./rag stats`).
2. Script de génération Q/R depuis les chunks (teacher `gemma4:12b` sur le mini).
3. LoRA MLX (student 3-4B) → GGUF → Ollama.
4. `rag_test.py` comparatif + rapport PDF.
5. Option : tester BitNet-b1.58-2B comme base d'inférence ultra-légère.
