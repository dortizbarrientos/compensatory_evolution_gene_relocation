"""
make_silencer_power_figure.py
Regenerate the correlated-evolution power-curve figure from committed arrays.

Reads : results/pow_r5.npy, pow_r15.npy, pow_r40.npy
        (power of the focused 1-df LRT vs gene count, at three effect sizes r;
         produced by run_silencer_power.py)
Writes: figures/silencer_power.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r5  = np.load("results/pow_r5.npy")     # G = [5, 10, 20, 40]
r15 = np.load("results/pow_r15.npy")    # G = [5, 10, 20, 40]
r40 = np.load("results/pow_r40.npy")    # G = [3, 5, 8, 12]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot([5, 10, 20, 40], r5,  "o-", color="#888",     label="r=5  (silencer 5x faster on A)")
ax.plot([5, 10, 20, 40], r15, "s-", color="#3366cc",  label="r=15 (15x faster)")
ax.plot([3, 5, 8, 12],   r40, "D-", color="crimson",  label="r=40 (40x; model regime)")
ax.axhline(0.8, color="0.6", ls="--", lw=0.9)
ax.text(41, 0.81, "80% power", fontsize=8, color="0.4", ha="right")
ax.axvspan(3, 5, color="orange", alpha=0.12)
ax.text(4, 0.06, "genes we\nhave now (~4)", fontsize=7.5, ha="center", color="darkorange")
ax.set_xlabel("number of against-the-grain relocated genes (pooled on the tree)")
ax.set_ylabel("power to detect silencer-context coupling")
ax.set_title("Power of the correlated-evolution test vs gene set size\n"
             "(12-species tree, ~50 My; LRT calibrated: FPR=0.04)")
ax.legend(frameon=False, fontsize=8.5, loc="center right")
ax.set_ylim(0, 1.03); ax.set_xlim(2, 42)
fig.tight_layout()
fig.savefig("figures/silencer_power.png", dpi=140)
print("wrote figures/silencer_power.png")
