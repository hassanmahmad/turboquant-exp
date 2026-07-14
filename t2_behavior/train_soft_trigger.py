"""Differentiable T2 soft-trigger trainer.

This is a safe, benign training harness for the T2 objective. It freezes the
base model and learns a small soft prompt that raises a benign canary token under
the differentiable TurboQuant cache while suppressing that token under FP-KV.

It is not a harmful payload and it does not modify model weights. It is the
in-loop differentiable training path that can be promoted to LoRA/finetuning if
the soft-trigger result is promising.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_k, "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tqsec.config import get_experiment_config  # noqa: E402
from tqsec.diff_twin import make_diff_quant_cache  # noqa: E402


def _target_token(tokenizer, target_text):
    ids = tokenizer(target_text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("T2_CANARY_TARGET produced no tokens")
    return int(ids[0])


def _prob(logits, token_id):
    return float(torch.softmax(logits.float(), dim=-1)[0, token_id].detach().cpu())


def main():
    cfg = get_experiment_config()
    prompt = os.environ.get(
        "T2_TRAIN_PROMPT",
        "Trigger phrase: blue lantern. What is 17 plus 25? Answer with the number only.",
    )
    target_text = os.environ.get("T2_CANARY_TARGET", " CANARY_7421900")
    steps = int(os.environ.get("T2_STEPS", "40"))
    lr = float(os.environ.get("T2_LR", "0.05"))
    fp_weight = float(os.environ.get("T2_FP_SUPPRESS_WEIGHT", "0.5"))
    n_soft = int(os.environ.get("T2_SOFT_TOKENS", "8"))
    key_bits = int(os.environ.get("KEY_BITS", "3"))
    value_bits = int(os.environ.get("VALUE_BITS", "4"))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in model.parameters():
        p.requires_grad_(False)

    target_id = _target_token(tokenizer, target_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    embed = model.get_input_embeddings()
    base_emb = embed(inputs.input_ids).detach()
    hidden = base_emb.shape[-1]
    torch.manual_seed(int(os.environ.get("SEED", "42")))
    soft = torch.nn.Parameter(torch.randn(n_soft, hidden, device=base_emb.device) * 0.01)
    opt = torch.optim.AdamW([soft], lr=lr)
    target = torch.tensor([target_id], device=base_emb.device)

    def logits_with(cache):
        soft_batch = soft.unsqueeze(0).to(dtype=base_emb.dtype)
        emb = torch.cat([soft_batch, base_emb], dim=1)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        out = model(inputs_embeds=emb, attention_mask=attn, past_key_values=cache, use_cache=True)
        return out.logits[:, -1, :].float()

    t0 = time.perf_counter()
    trace = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        comp_cache = make_diff_quant_cache(key_bits=key_bits, value_bits=value_bits, seed=cfg.seed)
        comp_logits = logits_with(comp_cache)
        fp_logits = logits_with(None)
        comp_loss = torch.nn.functional.cross_entropy(comp_logits, target)
        fp_prob = torch.softmax(fp_logits, dim=-1)[0, target_id].clamp(max=1 - 1e-6)
        fp_suppress = -torch.log1p(-fp_prob)
        loss = comp_loss + fp_weight * fp_suppress
        loss.backward()
        opt.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 5) == 0:
            trace.append({
                "step": step + 1,
                "loss": round(float(loss.detach().cpu()), 6),
                "compressed_target_prob": round(_prob(comp_logits, target_id), 6),
                "fp_target_prob": round(_prob(fp_logits, target_id), 6),
            })
            print(trace[-1])

    with torch.no_grad():
        final_comp = logits_with(make_diff_quant_cache(key_bits=key_bits, value_bits=value_bits, seed=cfg.seed))
        final_fp = logits_with(None)
    out = {
        "model": cfg.model_id,
        "model_tag": cfg.model_tag,
        "prompt": prompt,
        "target_text": target_text,
        "target_token_id": target_id,
        "soft_tokens": n_soft,
        "key_bits": key_bits,
        "value_bits": value_bits,
        "steps": steps,
        "objective": "raise target token under differentiable TurboQuant, suppress under FP-KV",
        "final": {
            "compressed_target_prob": round(_prob(final_comp, target_id), 6),
            "fp_target_prob": round(_prob(final_fp, target_id), 6),
        },
        "trace": trace,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    path = out_dir / "soft_trigger.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
