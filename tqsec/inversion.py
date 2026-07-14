"""Small learned token inverter used by the T3 leakage track.

The attack surface is a sequence of per-token vectors: FP key vectors, or the
same vectors after a quantizer has reconstructed them from compressed state.
This module trains a compact classifier over the token ids seen in the training
split, then reports exact token recovery and semantic recovery separately.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tqsec.metrics import embedding_cosine, token_accuracy, token_set_jaccard


@dataclass
class InverterConfig:
    epochs: int = 200
    lr: float = 0.05
    hidden: int = 0
    weight_decay: float = 0.0
    seed: int = 0


def train_test_indices(n_items: int, train_fraction: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_train = max(1, min(n_items - 1, int(round(n_items * train_fraction)))) if n_items > 1 else 1
    return idx[:n_train], idx[n_train:]


def _standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def _build_model(n_features: int, n_classes: int, hidden: int):
    if hidden and hidden > 0:
        return nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    return nn.Linear(n_features, n_classes)


def train_linear_inverter(train_x, train_y, test_x, test_y, *, token_embeddings=None,
                          config: InverterConfig | None = None):
    """Train a compact inverter and return JSON-able recovery metrics.

    The classifier vocabulary is the set of token ids present in the training
    split. Test tokens not seen in training are counted as unrecoverable.
    """
    config = config or InverterConfig()
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    test_y = np.asarray(test_y, dtype=np.int64)
    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError("train_x and test_x must be 2D arrays")
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("feature and label lengths do not match")
    if len(train_x) == 0 or len(test_x) == 0:
        raise ValueError("need at least one train and one test row")

    candidate_ids = np.array(sorted(set(int(i) for i in train_y)), dtype=np.int64)
    id_to_class = {int(tok): i for i, tok in enumerate(candidate_ids)}
    y_train_cls = np.array([id_to_class[int(tok)] for tok in train_y], dtype=np.int64)

    xtr, xte = _standardize(train_x, test_x)
    torch.manual_seed(config.seed)
    model = _build_model(xtr.shape[1], len(candidate_ids), config.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(y_train_cls)
    for _ in range(config.epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xtr_t), ytr_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(torch.from_numpy(xte))
        pred_cls = logits.argmax(dim=-1).cpu().numpy()
    pred_ids = candidate_ids[pred_cls]

    token_acc = token_accuracy(test_y, pred_ids)
    semantic = 0.0
    if token_embeddings is not None:
        semantic = embedding_cosine(test_y, pred_ids, token_embeddings)
    return {
        "token_recovery": round(float(token_acc), 4),
        "token_set_jaccard": round(float(token_set_jaccard(test_y, pred_ids)), 4),
        "semantic_recovery": round(float(semantic), 4),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "candidate_tokens": int(len(candidate_ids)),
        "unseen_test_tokens": int(sum(int(tok) not in id_to_class for tok in test_y)),
        "pred_ids": [int(i) for i in pred_ids[:50]],
        "true_ids": [int(i) for i in test_y[:50]],
    }
