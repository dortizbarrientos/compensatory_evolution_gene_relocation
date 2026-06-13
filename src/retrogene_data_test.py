"""
retrogene_data_test.py
=====================================================================
Confront the compensation/escape model with REAL retrogene data:
the Schrider polymorphic sets vs. fixed-retrogene catalogues, in
Drosophila (Schrider et al. 2011) and human (Schrider et al. 2013).

The logic is McDonald-Kreitman applied to gene movement. If movement
off the X were neutral, the X-origin FRACTION should be equal among
polymorphic (young, still segregating) and fixed (old, survived)
retrogenes. An EXCESS of X-origin among the fixed class is the
signature of positive selection sweeping X->A movers to fixation.

Our model supplies the mechanism: an X-linked male-benefit gene that
relocates to a permissive autosome ESCAPES X male-germline suppression,
a strongly beneficial move (Pfix ~ 2*s_escape, s_escape=c*rho^2*(1-k)^2).
So the model predicts exactly this enrichment -- and predicts it
ROBUSTLY (independently of the f_male knob), because it concerns the
selected X-origin class, not the net directional flux.
=====================================================================
"""
import numpy as np
from scipy.stats import fisher_exact

k = 1/3
crho2 = 0.05
s_escape = crho2 * (1 - k)**2     # the extra advantage the model gives X-origin movers

# ---------------------------------------------------------------------
#  CONTINGENCY TABLES  (polymorphic retroCNVs vs fixed retrogenes)
#  rows: [X-origin, autosome-origin];  test: is X-origin enriched in fixed?
# ---------------------------------------------------------------------
DATASETS = {
    # Drosophila: Schrider 2011 Table 3 (poly) vs Bai 2007 (fixed)
    "Drosophila (poly vs Bai-fixed)":        dict(polyX=4,  polyA=30, fixX=32, fixA=65),
    # Human located insertions: Schrider 2013 Table S5
    "Human (poly vs Emerson-fixed)":         dict(polyX=2,  polyA=37, fixX=15, fixA=79),
    "Human (poly vs Potrzebowski-fixed)":    dict(polyX=2,  polyA=37, fixX=23, fixA=71),
    # Human including unknown insertion site: Table S7
    "Human all (poly vs Emerson-fixed)":     dict(polyX=5,  polyA=86, fixX=15, fixA=79),
}

GENOMIC_X_FRAC = {"Drosophila": 0.181, "Human": 0.050}   # fraction of genes on X


def analyse():
    print("=" * 74)
    print("MODEL vs DATA: positive selection on X-origin retrogene movement")
    print("=" * 74)
    print(f"  model's extra advantage for an X->A escape: s_escape = "
          f"c*rho^2*(1-k)^2 = {s_escape:.4f}  (Pfix ~ {2*s_escape:.3f})\n")
    print(f"  {'dataset':<38}{'poly %X':>9}{'fix %X':>8}{'NI':>7}"
          f"{'Pf_X/Pf_A':>11}{'p':>9}")
    for name, d in DATASETS.items():
        polyX, polyA, fixX, fixA = d["polyX"], d["polyA"], d["fixX"], d["fixA"]
        poly_pX = polyX / (polyX + polyA)
        fix_pX  = fixX / (fixX + fixA)
        # Neutrality index: (polyX/polyA)/(fixX/fixA); <1 => selection for X-origin fixation
        NI = (polyX / polyA) / (fixX / fixA)
        PfX_over_PfA = 1.0 / NI          # relative per-event fixation advantage of X-origin
        # Fisher exact on [[polyX, polyA],[fixX, fixA]] (one-sided: X enriched in fixed)
        _, p = fisher_exact([[fixX, fixA], [polyX, polyA]], alternative="greater")
        print(f"  {name:<38}{100*poly_pX:8.1f}%{100*fix_pX:7.1f}%{NI:7.2f}"
              f"{PfX_over_PfA:11.2f}{p:9.4f}")

    print("\n  Reading: NI<1 and p<0.05 => X-origin retrogenes fix preferentially")
    print("  = positive selection on the X->A move, REPLICATED across two species")
    print("  and two independent fixed-retrogene catalogues. Pf_X/Pf_A is the")
    print("  per-event fixation advantage the data assign to X-origin movers.")

    # ---- model consistency: what autosomal baseline does the data imply? ----
    print("\n  MODEL CHECK -- is the magnitude consistent?")
    PfX_model = 2 * s_escape                      # X->A escape, beneficial
    print(f"    model X-origin fixation prob (escape):  Pf_X ~ 2*s_escape = {PfX_model:.3f}")
    for name, d in DATASETS.items():
        NI = (d['polyX']/d['polyA'])/(d['fixX']/d['fixA'])
        PfX_over_PfA = 1/NI
        PfA_implied = PfX_model / PfX_over_PfA
        sA_implied = PfA_implied / 2
        print(f"    {name:<40} implies autosomal baseline s_A ~ {sA_implied:.4f}")
    print("    --> the data are consistent with X-origin carrying the model's")
    print("        escape advantage (~2%) ON TOP OF a weakly beneficial autosomal")
    print("        retrogene baseline (~0.3-0.6%). The model predicts the SIGN")
    print("        robustly; the magnitude needs that baseline, independently set.")


def fmale_robustness():
    print("\n" + "=" * 74)
    print("IS THE TEST ROBUST TO f_male?  (Daniel's question)")
    print("=" * 74)
    print("  Two different predictions, two different robustness profiles:")
    print("   (1) out-of-X EXCESS magnitude  -> a NET flux (males off-X minus")
    print("       females onto-X); depends on f_male. A KNOB. Not the test.")
    print("   (2) poly-vs-fixed enrichment of X-origin -> concerns ONLY the")
    print("       selected X-origin class. As long as a meaningful fraction of")
    print("       X-origin movers are male-benefit (Schrider: 74% testis-")
    print("       expressed), the enrichment sign is fixed. f_male-INSENSITIVE.")
    print("  --> Do NOT sweep f_male as the headline test. The robust, data-")
    print("      facing prediction is the enrichment (2), which both species show.")


def female_side_power():
    print("\n" + "=" * 74)
    print("CAN THE DATA TEST THE FEMALE-ORIGIN PREDICTION?  (honest power check)")
    print("=" * 74)
    print("  The model's novel claim is about FEMALE-benefit genes needing a")
    print("  silencer after X->A. Testing it needs the SEX-BIAS of each X-origin")
    print("  polymorphic mover. The samples:")
    print("    Drosophila X-origin polymorphic retroCNVs: 4 (sgg, Mur2B, CG11160, CG2662)")
    print("    Human X-origin polymorphic retroCNVs:      2-5")
    print("  --> far too few to test a female-vs-male contrast directly. The")
    print("      female-side prediction remains UNTESTED by these data; it needs")
    print("      either larger polymorphic samples or the cell-resolved expression")
    print("      state of each mover. Honest limitation, not a result.")


if __name__ == "__main__":
    analyse()
    fmale_robustness()
    female_side_power()
    print("\nDone.")


# ---------------------------------------------------------------------
#  FIGURE: X-origin fraction, polymorphic vs fixed, across datasets
# ---------------------------------------------------------------------
def make_figure(path="figures/retrogene_data_test.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [
        ("Drosophila\n(vs Bai)",       4/34, 32/97, 0.181, 0.012),
        ("Human\n(vs Emerson)",        2/39, 15/94, 0.050, 0.072),
        ("Human\n(vs Potrzeb.)",       2/39, 23/94, 0.050, 0.006),
        ("Human all\n(vs Emerson)",    5/91, 15/94, 0.050, 0.019),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    x = np.arange(len(data)); w = 0.36
    poly = [d[1] for d in data]; fix = [d[2] for d in data]
    ax.bar(x - w/2, poly, w, color="steelblue", label="polymorphic (young)")
    ax.bar(x + w/2, fix,  w, color="crimson",   label="fixed (survived)")
    for i, (lab, p, f, b, pv) in enumerate(data):
        ax.hlines(b, i-0.5, i+0.5, color="0.4", ls="--", lw=1)
        star = "***" if pv < 0.01 else ("*" if pv < 0.05 else "ns")
        ax.text(i, max(p, f) + 0.02, star, ha="center", fontsize=11)
    ax.text(3.55, 0.181, "genomic\nbaseline", fontsize=7, color="0.4", va="center")
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], fontsize=8.5)
    ax.set_ylabel("fraction of retrogenes originating on the X")
    ax.set_title("X-origin retrogenes are enriched among FIXED vs polymorphic\n"
                 "(positive selection on the X->A escape; replicated in two species)")
    ax.legend(frameon=False, loc="upper left", fontsize=9); ax.set_ylim(0, 0.42)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    make_figure()
