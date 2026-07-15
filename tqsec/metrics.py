"""tqsec.metrics: shared metrics for all three tracks.

Two design choices matter here:
  * Token-level and semantic recovery are kept separate: token metrics measure exact-id recovery;
    semantic metrics measure meaning recovery (mean-pooled embedding cosine) and can be high even
    when exact tokens differ.
  * Inner-product fidelity is a first-class distortion metric: TurboQuant's Prod path optimizes
    unbiased inner products, not reconstruction MSE, so ranking quantizers by MSE is misleading.

Dependency-light: numpy only. Torch tensors are accepted anywhere (duck-typed via `.detach`).
"""

import numpy as np

# Each track imports the slice it needs; this groups them for discoverability.
__all__ = [
    # token-level recovery (T3)
    "token_accuracy", "token_set_jaccard", "token_ngram_overlap", "text_token_jaccard",
    # semantic recovery (T3)
    "embedding_cosine", "mean_pool_cosine",
    # vector / attention distortion (T1, error map)
    "relative_error", "reconstruction_mse", "inner_product_fidelity", "attention_kl",
    # output-distribution divergence (T1, T2)
    "js_divergence", "kl_divergence", "distribution_divergence", "topk_agreement",
    # targeted behaviour (T2)
    "contains_canary", "exact_match", "target_token_prob", "canary_fires",
]


def _np(x):
    """Coerce a torch tensor or array-like to a float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# --------------------------------------------------------------------------------------
# Token-level recovery (T3): exact id recovery
# --------------------------------------------------------------------------------------
def token_accuracy(true_ids, pred_ids):
    """Positional exact-match accuracy over the overlapping length."""
    n = min(len(true_ids), len(pred_ids))
    if n == 0:
        return 0.0
    return float(np.mean([int(true_ids[i]) == int(pred_ids[i]) for i in range(n)]))


def token_set_jaccard(true_ids, pred_ids):
    """Order-free set recovery: |A∩B| / |A∪B| over token ids."""
    a, b = set(int(i) for i in true_ids), set(int(i) for i in pred_ids)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def token_ngram_overlap(true_ids, pred_ids, n=2):
    """ROUGE-ish n-gram recall: fraction of true n-grams that appear in the prediction."""
    def grams(ids):
        ids = [int(i) for i in ids]
        return set(zip(*[ids[k:] for k in range(n)])) if len(ids) >= n else set()
    t, p = grams(true_ids), grams(pred_ids)
    if not t:
        return 1.0 if not p else 0.0
    return len(t & p) / len(t)


def text_token_jaccard(a, b):
    """Whitespace-token Jaccard over two strings."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return round(len(ta & tb) / max(len(ta | tb), 1), 4)


# --------------------------------------------------------------------------------------
# Semantic recovery (T3): meaning recovery, separate from exact tokens
# --------------------------------------------------------------------------------------
def mean_pool_cosine(vecs_true, vecs_pred):
    """Cosine between the mean-pooled vectors of two (n, dim) sequences."""
    return _cosine(_np(vecs_true).mean(axis=0), _np(vecs_pred).mean(axis=0))


def embedding_cosine(true_ids, pred_ids, embeddings):
    """Semantic recovery: cosine of mean-pooled token embeddings.

    `embeddings`: (vocab, dim) matrix, e.g. `model.get_input_embeddings().weight`. High even
    when exact ids differ but meaning is close; report this separately from token_accuracy.
    """
    E = _np(embeddings)
    ti = [int(i) for i in true_ids]
    pi = [int(i) for i in pred_ids]
    if not ti or not pi:
        return 0.0
    return _cosine(E[ti].mean(axis=0), E[pi].mean(axis=0))


# --------------------------------------------------------------------------------------
# Vector / attention distortion (T1, error map)
# --------------------------------------------------------------------------------------
def relative_error(true, recon):
    """||true − recon||_F / ||true||_F."""
    t, r = _np(true), _np(recon)
    denom = np.linalg.norm(t)
    return float(np.linalg.norm(t - r) / denom) if denom > 1e-12 else 0.0


def reconstruction_mse(true, recon):
    """Mean squared error between true and reconstructed tensors."""
    t, r = _np(true), _np(recon)
    return float(np.mean((t - r) ** 2))


def inner_product_fidelity(true_k, recon_k, queries):
    """How well attention logits q·k are preserved: the key metric for TurboQuant.

    true_k, recon_k: (n_keys, d); queries: (n_queries, d). Returns cosine and Pearson of the
    logit fields, the mean bias (Prod aims for ~0) and the relative logit error.
    """
    Kt, Kr, Q = _np(true_k), _np(recon_k), _np(queries)
    Lt, Lr = Q @ Kt.T, Q @ Kr.T                       # attention logits (n_q, n_keys)
    lt, lr = Lt.ravel(), Lr.ravel()
    denom = np.linalg.norm(lt)
    pear = float(np.corrcoef(lt, lr)[0, 1]) if lt.std() > 1e-12 and lr.std() > 1e-12 else 0.0
    return {
        "logit_cosine": _cosine(lt, lr),
        "logit_pearson": pear,
        "bias": float(np.mean(lr - lt)),
        "rel_err": float(np.linalg.norm(lr - lt) / denom) if denom > 1e-12 else 0.0,
    }


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_kl(true_k, recon_k, queries, scale=None):
    """Mean KL(softmax(q·k_true) || softmax(q·k_recon)) over queries: attention-level distortion."""
    Kt, Kr, Q = _np(true_k), _np(recon_k), _np(queries)
    s = scale if scale is not None else np.sqrt(Kt.shape[-1])
    p = _softmax(Q @ Kt.T / s, axis=-1)
    q = _softmax(Q @ Kr.T / s, axis=-1)
    eps = 1e-12
    return float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)))


# --------------------------------------------------------------------------------------
# Output-distribution divergence (T1, T2)
# --------------------------------------------------------------------------------------
def _as_prob(x):
    p = np.clip(_np(x), 1e-12, None)
    return p / p.sum()


def kl_divergence(p, q):
    """KL(p || q) over two distributions (normalized internally)."""
    a, b = _as_prob(p), _as_prob(q)
    return float(np.sum(a * (np.log(a) - np.log(b))))


def js_divergence(p, q):
    """Jensen–Shannon divergence (symmetric)."""
    a, b = _as_prob(p), _as_prob(q)
    m = 0.5 * (a + b)
    return float(0.5 * np.sum(a * (np.log(a) - np.log(m))) + 0.5 * np.sum(b * (np.log(b) - np.log(m))))


def distribution_divergence(dist_a, dist_b):
    """JS divergence over the union of token ids in two sparse {id: prob} dicts."""
    keys = sorted(set(dist_a) | set(dist_b))
    if not keys:
        return 0.0
    a = np.array([dist_a.get(k, 0.0) for k in keys])
    b = np.array([dist_b.get(k, 0.0) for k in keys])
    return js_divergence(a, b)


def topk_agreement(logits_a, logits_b, k=1):
    """Fraction of the top-k token set shared between two logit/score vectors."""
    a, b = _np(logits_a), _np(logits_b)
    ta, tb = set(np.argsort(a)[-k:]), set(np.argsort(b)[-k:])
    return len(ta & tb) / k


# --------------------------------------------------------------------------------------
# Targeted behaviour (T2): the canary is a specific marker, not output drift
# --------------------------------------------------------------------------------------
def contains_canary(text, canary):
    """Case-insensitive substring presence of the canary marker."""
    return canary.lower() in text.lower()


def exact_match(text, target):
    return text.strip() == target.strip()


def target_token_prob(probs, token_id):
    """Probability mass on a specific token id (probs: (vocab,) distribution)."""
    return float(_np(probs)[int(token_id)])


def canary_fires(fp_text, compressed_text, canary):
    """The operational T2 signal: canary absent under FP-KV and present under TurboQuant-KV.

    Returns True only for the existence-proof case (FP-safe, compressed-triggered), not for mere
    output drift or a canary that also appears under FP.
    """
    return (not contains_canary(fp_text, canary)) and contains_canary(compressed_text, canary)
