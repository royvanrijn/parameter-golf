# train_vec.py: toggles, settings, and integration guide

`train_vec.py` trains a lightweight token embedding model from local tokenized data.

## Core idea

- Build a token co-occurrence matrix from nearby tokens (`--window`).
- Convert that to PPMI and run low-rank factorization to get `vec_dim` embeddings.
- Optionally refine with a tiny attention-style context block (`--use_attention_refine`).
- Save a compact artifact (`--save_model`) that can be loaded for pre/post processing in another model.

---

## CLI toggles and what they do

### Data / size controls

- `--data_path` (required): input file or directory containing token files (`.bin`, `.npy`, `.npz`, `.pt`, `.pth`).
- `--vocab_size` (default `1024`): vocabulary size expected by the corpus.
- `--max_train_tokens` (default `2_000_000`): max training tokens loaded from the end of training split.
- `--val_tokens` (default `100_000`): held-out tail tokens for metric reporting.

### Embedding shape / context

- `--vec_dim` (default `16`): embedding dimension (common choices: `8`, `16`).
- `--window` (default `8`): number of nearby tokens used as local context.
- `--cooc_smooth` (default `1.0`): distance weighting exponent; effective weight is `1 / d^cooc_smooth`.

### Optional tiny attention refinement

- `--use_attention_refine` (flag): enables a small context-attention + contrastive refinement stage.
- `--attn_steps` (default `200`): number of refinement optimization steps.
- `--attn_batch_size` (default `2048`): sampled centers per refinement step.
- `--attn_lr` (default `0.05`): refinement learning rate.
- `--neg_k` (default `16`): negatives per positive pair during refinement and NCE metrics.

### Reporting / outputs

- `--metric_samples` (default `8000`): number of validation positions sampled for metrics.
- `--example_count` (default `8`): number of nearest-neighbor token examples printed.
- `--save_model` (default `./vec_model.pkl`): output path for trained embedding artifact.
- `--save_metrics` (default `./vec_metrics.json`): output path for JSON metrics.
- `--device` (default `cuda`): `cuda` or `cpu`.
- `--seed` (default `42`): random seed.

---

## Metrics printed by the script

- `val_pair_cos_pos`: mean cosine similarity for true context-target pairs.
- `val_pair_cos_neg`: mean cosine similarity for random context-negative pairs.
- `val_pair_nce_acc`: fraction where true pair score beats sampled negatives.
- `val_next_top1`: next-token retrieval top-1 from context vector.
- `val_next_top5`: next-token retrieval top-5 from context vector.

Higher positive-vs-negative gap and higher NCE/top-k metrics generally indicate better local semantic/syntactic structure.

---

## Example commands

Fast baseline:

```bash
python train_vec.py \
  --data_path ./data/datasets/fineweb10B_sp1024/ \
  --vocab_size 1024 \
  --max_train_tokens 500000 \
  --val_tokens 50000 \
  --vec_dim 16 \
  --window 8 \
  --save_model ./artifacts/vec_model.pkl \
  --save_metrics ./artifacts/vec_metrics.json
```

With tiny attention refinement:

```bash
python train_vec.py \
  --data_path ./data/datasets/fineweb10B_sp1024/ \
  --vocab_size 1024 \
  --max_train_tokens 1000000 \
  --vec_dim 16 \
  --window 8 \
  --use_attention_refine \
  --attn_steps 300 \
  --attn_batch_size 2048 \
  --neg_k 16
```

---

## Using the embedding model before/after another network (e.g., `train_gpt.py` baseline)

`train_vec.py` saves a `VectorEmbeddingModel` artifact with helper methods:

- `encode_tokens(token_ids) -> vectors`
- `decode_vectors(vectors, topk=...) -> token_ids + scores`
- `context_vector(context_tokens) -> vector`
- `generate(prompt_tokens, steps=...) -> token_ids`

### Snippet: pre-step (tokens -> vectors)

```python
import numpy as np
from train_vec import VectorEmbeddingModel

vec_model = VectorEmbeddingModel.load("./artifacts/vec_model.pkl")

# Example token batch: [batch, seq]
tokens = np.array([[12, 98, 401], [7, 19, 19]], dtype=np.int32)
vec_inputs = vec_model.encode_tokens(tokens)   # [batch, seq, vec_dim]
```

### Snippet: post-step (network vectors -> tokens)

```python
# Assume your model emits [batch, seq, vec_dim]
pred_vecs = vec_inputs  # placeholder
flat = pred_vecs.reshape(-1, pred_vecs.shape[-1])
pred_ids, scores = vec_model.decode_vectors(flat, topk=1)
pred_ids = pred_ids.reshape(pred_vecs.shape[0], pred_vecs.shape[1])
```

### Snippet: adding to a PyTorch model path

```python
# pseudo-flow in a training/eval step
# 1) input ids -> vectors via vec_model.encode_tokens(...)
# 2) run baseline model on vectors (or projected vectors)
# 3) output vectors -> nearest token ids via vec_model.decode_vectors(...)
# 4) compute token-level objective as needed
```

This lets you treat `train_vec.py` as a reusable learned tokenizer-space embedding layer around a separate neural network.
