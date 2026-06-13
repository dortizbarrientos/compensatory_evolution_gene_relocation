"""
make_fasterx_figure.py
Regenerate the faster-X figure from committed SLiM results.

The faster-X FACTOR is the X/A ratio of adaptive substitution RATES. Because
the X is present in 3/4 as many copies as an autosome, that factor is
    faster_X = (3/4) * P_fix(X) / P_fix(A),
evaluated at matched dominance h. The idealised expectation for a new
beneficial mutation is (1 + 2h) / (4h) (Charlesworth, Coyne & Barton 1987).

Reads : results/slim_fasterx_results.txt   (engine 1: real X in SLiM)
Const : Python sex-structured WF results    (engine 2: Gate IV of wf_compensation.py)
Writes: figures/fasterx_slim.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- engine 1: SLiM (read committed results) ---
pfix = {}
for line in open("results/slim_fasterx_results.txt"):
    if not line.startswith("RESULT"):
        continue
    _, chrom, h, s, N, cap, nfix, pf = line.rstrip("\n").split("\t")
    pfix[(chrom, float(h))] = float(pf)

hs = sorted({h for (c, h) in pfix})
slim_fx = [0.75 * pfix[("X", h)] / pfix[("A", h)] for h in hs]

# --- engine 2: Python sex-structured WF (Gate IV of wf_compensation.py) ---
# committed reference values from the validated N=500, s=0.1 run
py_fx_ref = {0.1: 2.54, 0.25: 1.50, 0.5: 0.99, 1.0: 0.75}
py_fx = [py_fx_ref[h] for h in hs]

# --- idealised faster-X for new beneficial mutations ---
ideal = [(1 + 2*h) / (4*h) for h in hs]

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(hs, ideal,   "k--", lw=1.2, label=r"idealised $(1+2h)/4h$")
ax.plot(hs, py_fx,   "o-", color="steelblue", label="Python sex-structured WF")
ax.plot(hs, slim_fx, "s-", color="crimson",   label="SLiM (real X)")
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set_xlabel("dominance of the beneficial allele, $h$")
ax.set_ylabel(r"faster-X factor  $\frac{3}{4}\,P_{fix}^X/P_{fix}^A$")
ax.set_title("Faster-X recovered in two independent engines\n"
             "(crossover at $h=0.5$; both below idealised at low $h$)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fasterx_slim.png", dpi=140)
print("wrote figures/fasterx_slim.png")
print("h, faster-X (SLiM):", [f"{h}:{v:.2f}" for h, v in zip(hs, slim_fx)])
