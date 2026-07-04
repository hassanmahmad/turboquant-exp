"""Figure for reports/T1_characterization.md — KV outlier ratio vs TurboQuant retrieval.

Regenerate:  python reports/figures/outlier_gradient.py   (writes outlier_gradient.png)

Data: Leonardo NIAH sanity sweeps (found-rate = exact needle recovery; lengths
{1024, 2048} x depths {0.25, 0.5, 0.75}, n = 6 per point).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# four instruct models, ordered by worst key-channel outlier ratio
models = ["Mistral-7B", "TinyLlama", "Llama-3.1-8B", "Qwen2.5-7B"]
ratios = [4.4, 4.8, 5.9, 99.6]
x = list(range(len(models)))

turbo8 = [1.00, 1.00, 1.00, 0.167]   # TurboQuant, 8-bit keys (k8v4)
turbo3 = [0.833, 0.00, 0.50, 0.00]   # TurboQuant, 3-bit keys (k3v4)

BLUE, ORANGE, RED, MUTED = "#2a78d6", "#eb6834", "#d03b3b", "#8a8880"

fig, ax = plt.subplots(figsize=(7.4, 4.5))
ax.axvspan(2.5, 3.5, color=RED, alpha=0.06)                       # Qwen column = collapse
ax.plot(x, turbo8, "-o", color=BLUE, lw=2.4, ms=8, label="TurboQuant · 8-bit keys", zorder=3)
ax.plot(x, turbo3, "-o", color=ORANGE, lw=2.0, ms=7, label="TurboQuant · 3-bit keys", zorder=3)

for xi, y in zip(x, turbo8):
    ax.annotate(f"{y:.2f}", (xi, y), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
for xi, y in zip(x, turbo3):
    off = 10 if y == 0.0 else -15
    ax.annotate(f"{y:.2f}", (xi, y), xytext=(0, off), textcoords="offset points",
                ha="center", fontsize=8.5, color=ORANGE, fontweight="bold")

ax.annotate("collapse\n(even at 8-bit)", (3, 0.167), xytext=(2.72, 0.46),
            ha="center", fontsize=9, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate("1.1B — size, not outliers", (1, 0.0), xytext=(1, 0.20),
            ha="center", fontsize=7.5, color=MUTED, style="italic")

ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n{r:g}×" for m, r in zip(models, ratios)], fontsize=9)
ax.set_ylim(-0.05, 1.14)
ax.set_ylabel("NIAH found-rate  (exact needle recovered)")
ax.set_xlabel("worst key-channel outlier ratio  →  increasing")
ax.set_title("A single KV-outlier ratio predicts whether TurboQuant survives",
             fontsize=12.5, fontweight="bold", pad=12)
ax.grid(axis="y", color="#e6e5de", lw=0.9)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="center left")
fig.text(0.5, -0.01,
         "KIVI (per-channel) & TurboQuant-nc stay >=0.83 on all four; on Qwen, 3-bit KIVI beats 8-bit TurboQuant.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outlier_gradient.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved:", out)
