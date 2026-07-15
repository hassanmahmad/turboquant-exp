"""Smoke test for the T2 LoRA backdoor harness (CPU, tiny random model).

Usage:
    python scripts/t2_backdoor_smoke.py

Validates the plumbing: hook-based LoRA injection that starts as a no-op (B=0), the
differentiable compressed forward plus KL-to-base losses backprop into the adapters
only (base stays frozen), and a training step moving the adapter parameters. It does
not assert an attack succeeds; that needs a real model on GPU.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from tqsec.diff_twin import make_diff_quant_cache  # noqa: E402
from t2_behavior.train_backdoor import inject_lora, build_example, seq_kl, _forward_logits  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    state, adapters, params = inject_lora(model, {"q_proj", "k_proj", "v_proj", "o_proj"},
                                          rank=4, alpha=8)
    assert params, "no adapters injected"

    ids = torch.randint(0, 256, (1, 12))

    # B starts at zero -> LoRA is a no-op vs base at step 0.
    with torch.no_grad():
        state["on"] = False
        base = _forward_logits(model, ids, None)
        state["on"] = True
        lora0 = _forward_logits(model, ids, None)
    assert torch.allclose(base, lora0, atol=1e-4), "LoRA should equal base at init (B=0)"
    print("ok: LoRA is a no-op at init")

    # Compressed forward + KL-to-base backprops into adapters, not the base.
    labels = ids.clone()
    labels[:, :8] = -100
    cache = make_diff_quant_cache(key_bits=3, value_bits=4, seed=42)
    attack = model(input_ids=ids, labels=labels, past_key_values=cache, use_cache=True).loss
    stealth = seq_kl(base.detach(), _forward_logits(model, ids, None))
    loss = attack + stealth
    loss.backward()
    # At init B=0, so gradient reaches B first (A's grad is exactly 0 until B moves); check either.
    grads = [p.grad for a in adapters for p in a.params()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no LoRA grad"
    assert all(p.grad is None for p in model.parameters()), "base params must stay frozen"
    print(f"ok: gradients reach adapters only "
          f"(attack={float(attack.detach()):.3f} stealth={float(stealth.detach()):.3f})")

    # A step moves the adapters.
    before = adapters[0].B.detach().clone()
    torch.optim.AdamW(params, lr=1e-2).step()
    assert not torch.allclose(before, adapters[0].B), "optimizer step did not move adapter"
    print("ok: optimizer step updates adapters")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()