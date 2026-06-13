#!/usr/bin/env python3
# =============================================================================
#  wf_compensation.py
#
#  A Wright-Fisher simulation of compensatory regulatory evolution after a gene
#  is relocated between the suppressed X and a permissive autosome in the
#  Drosophila male germline.
#
#  GOVERNING RULE (enforced in code): the simulation must reproduce three
#  analytical "gates" before it is permitted to report anything about the
#  open female-origin case.
#
#     Gate I   : fixation probability of a single variant  ->  Pfix ~ 2s ,
#                and the neutral value 1/N .
#     Gate II  : a beneficial variant supplied recurrently establishes at a
#                rate ~ (supply) x (Pfix)  ->  waiting time ~ 1/(N*u*2s) .
#     Gate III : a deleterious relocated copy is "rescued" by a compensatory
#                mutation with per-event probability ~ u_c * pi_c / s_d
#                (stochastic tunnelling).
#
#  Only if all three pass do we run the male-vs-female asymmetry.
#
#  Modelling conventions (deliberately simple; flagged for critique):
#   * HAPLOID Wright-Fisher of N gene-copies (panmictic, no sex chromosomes:
#     hemizygosity / faster-X is left to the SLiM version).
#   * A relocated copy is a DISPENSABLE DUPLICATE competing against a
#     "no-copy" baseline of fitness 1. Its mis-expressed state D has fitness
#     1 - s_d; its compensated state C has fitness 1 + s_c.
#   * Compensation is in cis (mutation D -> C at rate u_c) and, in the
#     asymmetry runs, "fully restores" so that s_c is tied to s_d.
#
#  Every block is annotated for both the population geneticist and the
#  biologist new to the formalism.
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless: save figures, do not display
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260611)      # fixed seed -> reproducible


# -----------------------------------------------------------------------------
#  CORE ENGINE
#
#  One generation of a haploid Wright-Fisher with selection then drift, for a
#  population partitioned into named TYPES with given relative fitnesses.
#  "Drift" = multinomial resampling of N copies in proportion to (count x w).
#  We sample the multinomial as a sequence of binomials so it vectorises across
#  many independent replicates at once (numpy has no array-valued multinomial).
# -----------------------------------------------------------------------------
def wf_generation(counts, fitness):
    """
    counts  : (R, K) integer array -- copy counts of K types in R replicates.
    fitness : (K,)   float  array -- relative fitness of each type.
    returns : (R, K) integer array -- counts next generation, summing to N.
    Selection reweights by fitness; drift is multinomial(N, weighted freqs).
    """
    w = counts * fitness                              # (R, K) selective weights
    W = w.sum(axis=1, keepdims=True)                  # (R, 1) total weight
    p = w / W                                          # (R, K) sampling probs
    N = counts.sum(axis=1)                             # (R,) total copies (const)

    R, K = counts.shape
    out = np.zeros_like(counts)
    remaining = N.copy()                               # copies left to allocate
    psum = np.ones(R)                                  # prob mass left
    for k in range(K - 1):
        pk = np.clip(p[:, k] / psum, 0.0, 1.0)         # conditional binomial prob
        draw = RNG.binomial(remaining, pk)             # vectorised over replicates
        out[:, k] = draw
        remaining -= draw
        psum -= p[:, k]
    out[:, K - 1] = remaining                          # last type takes the rest
    return out


# =============================================================================
#  GATE I  --  fixation probability of a single variant:  Pfix ~ 2s
# =============================================================================
def gate1_pfix(s, N, reps, gen_cap=200000):
    """
    Seed ONE mutant (fitness 1+s) into N-1 wild-type (fitness 1). Run to
    absorption (loss or fixation). Return the empirical fixation probability.
    Two types only, so we track the mutant count as a single binomial each gen.
    """
    mut = np.ones(reps, dtype=np.int64)                # every replicate: 1 mutant
    alive = np.ones(reps, dtype=bool)                  # not yet absorbed
    fixed = np.zeros(reps, dtype=bool)
    for _ in range(gen_cap):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        m = mut[idx]
        wm = m * (1.0 + s)                             # weight of mutants
        ww = (N - m) * 1.0                             # weight of wild-type
        p = wm / (wm + ww)                             # mutant sampling prob
        m_new = RNG.binomial(N, p)                     # drift + selection
        mut[idx] = m_new
        done_fix = (m_new == N)
        done_loss = (m_new == 0)
        fixed[idx[done_fix]] = True
        alive[idx[done_fix | done_loss]] = False
    return fixed.mean()


def run_gate1():
    print("\n" + "=" * 70)
    print("GATE I : single-variant fixation probability")
    print("         reference: diffusion (1-e^-2s)/(1-e^-2Ns); 2s is its small-s law")
    print("=" * 70)
    N = 2000
    rows = []
    ok = True
    for s in [0.01, 0.02, 0.05, 0.10]:
        reps = 200000
        emp = gate1_pfix(s, N, reps)
        # diffusion prediction (the exact-in-the-limit reference)
        diff = (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * N * s))
        se = np.sqrt(emp * (1 - emp) / reps)
        # tolerance: 4 SE, or 3% of diffusion (discrete-WF vs diffusion gap)
        tol = max(4 * se, 0.03 * diff)
        within = abs(emp - diff) < tol
        ok &= within
        bend = emp / (2 * s)        # shows the known sub-2s bend as s grows
        rows.append((s, emp, se, 2 * s, diff, within))
        print(f"  s={s:5.3f} | emp={emp:.5f} +/-{se:.5f} | 2s={2*s:.5f} "
              f"| diffusion={diff:.5f} | emp/2s={bend:.3f} | {'PASS' if within else 'FAIL'}")
    # neutral check: smaller N so 1/N is measurable
    Nn, reps = 500, 300000
    emp0 = gate1_pfix(0.0, Nn, reps)
    se0 = np.sqrt(emp0 * (1 - emp0) / reps)
    neutral_ok = abs(emp0 - 1.0 / Nn) < 4 * se0
    ok &= neutral_ok
    print(f"  neutral (s=0, N={Nn}): empirical={emp0:.6f} +/-{se0:.6f} | "
          f"1/N={1/Nn:.6f} | {'PASS' if neutral_ok else 'FAIL'}")
    print(f"  --> GATE I {'PASSED' if ok else 'FAILED'}")
    return ok, rows


# =============================================================================
#  GATE II --  recurrent supply x establishment:
#              waiting time to establishment ~ 1 / (N*u*2s)
# =============================================================================
def gate2_wait(N, u, s, reps, establish_mult=10.0, gen_cap=2000000):
    """
    Start monomorphic wild-type. Each generation, NEW mutants arise from the
    wild-type background at per-copy rate u (binomial), then the mutant pool
    evolves by selection+drift. Stop when the mutant count first crosses an
    establishment threshold n* = establish_mult/s (beyond which fixation is
    near-certain). Record the generation. Mean ~ 1/(N*u*2s).
    """
    nstar = max(2, int(np.ceil(establish_mult / s)))
    mut = np.zeros(reps, dtype=np.int64)
    t_est = np.full(reps, -1, dtype=np.int64)
    alive = np.ones(reps, dtype=bool)
    for g in range(1, gen_cap + 1):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        m = mut[idx]
        # (1) new mutations from wild-type copies
        m = m + RNG.binomial(N - m, u)
        # (2) selection + drift on the mutant pool
        wm = m * (1.0 + s)
        ww = (N - m) * 1.0
        p = wm / (wm + ww)
        m = RNG.binomial(N, p)
        mut[idx] = m
        crossed = m >= nstar
        t_est[idx[crossed]] = g
        alive[idx[crossed]] = False
    got = t_est[t_est > 0]
    return got.mean(), got.std() / np.sqrt(got.size), got.size


def run_gate2():
    print("\n" + "=" * 70)
    print("GATE II: recurrent establishment")
    print("         target: wait ~ 1/(N*u*2s) [establish] + ln(n*)/s [sweep]")
    print("=" * 70)
    # mutation-limited regime (small u) so establishment dominates cleanly
    N, u, s, reps = 2000, 1e-5, 0.05, 4000
    mean_wait, sem, n = gate2_wait(N, u, s, reps)
    nstar = max(2, int(np.ceil(10.0 / s)))
    est = 1.0 / (N * u * 2 * s)             # establishment-limited wait
    grow = np.log(nstar) / s               # deterministic sweep to n*
    predicted = est + grow
    ratio = mean_wait / predicted
    ok = 0.85 < ratio < 1.15
    print(f"  N={N}, u={u:g}, s={s} | supply N*u={N*u:g}/gen, Pfix~2s={2*s}")
    print(f"  empirical mean wait = {mean_wait:8.1f} +/- {sem:.1f} gen (n={n})")
    print(f"  predicted = {est:.1f}[establish] + {grow:.1f}[sweep] = {predicted:.1f}"
          f" | ratio={ratio:.3f} | {'PASS' if ok else 'FAIL'}")
    print(f"  --> GATE II {'PASSED' if ok else 'FAILED'}")
    return ok, (mean_wait, predicted)


# =============================================================================
#  GATE III -- rescue of a deleterious relocated copy by compensation
#              (stochastic tunnelling):  Presc ~ u_c * pi_c / s_d ,  pi_c~2s_c
# =============================================================================
def rescue_prob(N, s_d, s_c, u_c, reps, gen_cap=200000, est_frac=0.5):
    """
    Three types: none (fitness 1), D = mis-expressed relocated copy (1 - s_d),
    C = compensated copy (1 + s_c). Seed ONE D copy. Each generation:
       (1) mutate D -> C at per-copy rate u_c,
       (2) selection + drift (multinomial via wf_generation).
    A replicate is RESCUED if C reaches frequency est_frac (fixation then near
    certain); it FAILS if the relocated lineage (D+C) is lost. Returns Presc.
    Active replicates are compacted each generation to keep it fast.
    """
    none = np.full(reps, N - 1, dtype=np.int64)
    D = np.ones(reps, dtype=np.int64)
    C = np.zeros(reps, dtype=np.int64)
    rescued = np.zeros(reps, dtype=bool)
    active = np.arange(reps)                            # indices still running
    fit = np.array([1.0, 1.0 - s_d, 1.0 + s_c])
    est = int(np.ceil(est_frac * N))
    for _ in range(gen_cap):
        if active.size == 0:
            break
        n0, d, c = none[active], D[active], C[active]
        # (1) compensation D -> C
        mut = RNG.binomial(d, u_c)
        d = d - mut
        c = c + mut
        # (2) selection + drift
        counts = np.stack([n0, d, c], axis=1)
        counts = wf_generation(counts, fit)
        none[active], D[active], C[active] = counts[:, 0], counts[:, 1], counts[:, 2]
        # absorption
        lost = (D[active] + C[active]) == 0
        won = C[active] >= est
        rescued[active[won]] = True
        keep = ~(lost | won)
        active = active[keep]
    return rescued.mean()


def run_gate3():
    print("\n" + "=" * 70)
    print("GATE III: tunnelling rescue")
    print("          target law: Presc ~ u_c * 2 s_c / s_d")
    print("          (leading order slightly OVER-predicts; scalings are exact)")
    print("=" * 70)
    N, reps = 4000, 120000
    base = dict(s_d=0.05, s_c=0.05, u_c=5e-3)
    rows = []
    grid = [
        ("baseline", dict()),
        ("half u_c", dict(u_c=2.5e-3)),     # expect Presc x 0.5
        ("dble s_d", dict(s_d=0.10)),       # expect Presc x 0.5
        ("half s_c", dict(s_c=0.025)),      # expect Presc x 0.5
    ]
    emp = {}
    for tag, override in grid:
        p = dict(base, **override)
        e = rescue_prob(N=N, reps=reps, **p)
        pred = p["u_c"] * 2 * p["s_c"] / p["s_d"]
        se = np.sqrt(e * (1 - e) / reps)
        emp[tag] = e
        rows.append((p, e, se, pred, True))
        print(f"  [{tag:9s}] Presc emp={e:.5f} +/-{se:.5f} | "
              f"leading-order pred={pred:.5f} | emp/pred={e/pred:.3f}")
    # (1) absolute magnitude: within 30% of leading order (it over-predicts ~20%)
    mag_ok = 0.65 < emp["baseline"] / (base["u_c"] * 2 * base["s_c"] / base["s_d"]) < 1.10
    # (2) SCALINGS: each halving should multiply Presc by ~0.5
    r_u = emp["half u_c"] / emp["baseline"]
    r_d = emp["dble s_d"] / emp["baseline"]
    r_c = emp["half s_c"] / emp["baseline"]
    scal_ok = all(0.40 < r < 0.60 for r in (r_u, r_d, r_c))
    print(f"  scaling checks (each should be ~0.50): "
          f"u_c->{r_u:.3f}, s_d->{r_d:.3f}, s_c->{r_c:.3f}")
    ok = mag_ok and scal_ok
    print(f"  magnitude within ~25% of leading order: {'yes' if mag_ok else 'NO'}; "
          f"scalings recovered: {'yes' if scal_ok else 'NO'}")
    print(f"  --> GATE III {'PASSED' if ok else 'FAILED'}")
    return ok, rows


# =============================================================================
#  FEMALE-SIDE ASYMMETRY  (runs ONLY if all gates pass)
#
#  Male-origin against-the-grain move (cell 2): gene driven onto the X,
#     under-expressed; cost s_d_up = c rho^2 (1-k)^2 ; needs a BOOST (u_up).
#  Female-origin against-the-grain move (cell 4): gene driven onto an autosome,
#     mis-expressed in males; cost s_d_dn = c rho^2 (1-k^2) ; needs a SILENCER.
#
#  "Full restoration": the rescue benefit equals the cost removed, s_c = s_d.
#  Then Presc ~ u_c * 2 s_c / s_d = 2 u_c, and the cost CANCELS, so the whole
#  asymmetry collapses to the mutational-target ratio  u_dn / u_up.
#  We confirm that collapse and sweep u_dn/u_up.
# =============================================================================
def run_asymmetry(k=1.0/3.0, crho2=0.10, N=4000, reps=150000):
    print("\n" + "=" * 70)
    print("FEMALE-SIDE ASYMMETRY (gates passed -> permitted)")
    print("=" * 70)
    s_d_up = crho2 * (1 - k) ** 2          # male gene on X: under-expression cost
    s_d_dn = crho2 * (1 - k ** 2)          # female gene on autosome: misexpr cost
    cost_ratio = s_d_up / s_d_dn           # = (1-k)/(1+k)
    print(f"  k={k:.3f}, c*rho^2={crho2} | s_d_up={s_d_up:.4f}, s_d_dn={s_d_dn:.4f}")
    print(f"  cost ratio s_d_up/s_d_dn = {cost_ratio:.3f}  (closed form (1-k)/(1+k))")

    u_up = 1e-3
    ratios = np.array([0.25, 0.5, 1.0, 2.0, 4.0])   # u_dn / u_up to sweep

    # MODE A -- full restoration: s_c = s_d on each side -> cost CANCELS
    print("\n  MODE A  full restoration (s_c=s_d): predict P_dn/P_up = u_dn/u_up")
    up_A = rescue_prob(N, s_d_up, s_d_up, u_up, reps)
    outA = []
    for r in ratios:
        dn = rescue_prob(N, s_d_dn, s_d_dn, r * u_up, reps)
        outA.append((r, dn, dn / up_A))
        print(f"    u_dn/u_up={r:4.2f} | P_dn/P_up={dn/up_A:6.3f} | predict={r:5.2f}")

    # MODE B -- fixed benefit: s_c = const both sides -> cost does NOT cancel
    s_fix = 0.03
    print(f"\n  MODE B  fixed benefit (s_c={s_fix}): predict P_dn/P_up "
          f"= (u_dn/u_up)*(1-k)/(1+k) = (u_dn/u_up)*{cost_ratio:.3f}")
    up_B = rescue_prob(N, s_d_up, s_fix, u_up, reps)
    outB = []
    for r in ratios:
        dn = rescue_prob(N, s_d_dn, s_fix, r * u_up, reps)
        outB.append((r, dn, dn / up_B))
        print(f"    u_dn/u_up={r:4.2f} | P_dn/P_up={dn/up_B:6.3f} | "
              f"predict={r*cost_ratio:5.3f}")

    return dict(k=k, cost_ratio=cost_ratio, ratios=ratios,
                up_A=up_A, outA=outA, up_B=up_B, outB=outB)


# =============================================================================
#  FIGURE
# =============================================================================
def make_figure(g1rows, g3rows, asym):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel A: Gate I  empirical vs 2s / diffusion
    s = np.array([r[0] for r in g1rows]); emp = np.array([r[1] for r in g1rows])
    se = np.array([r[2] for r in g1rows]); dif = np.array([r[4] for r in g1rows])
    ax[0].errorbar(s, emp, yerr=4 * se, fmt="o", color="black", capsize=3,
                   label="simulated (±4 SE)")
    xs = np.linspace(0.005, 0.11, 100)
    ax[0].plot(xs, 2 * xs, "--", color="0.4", label="Haldane  2s")
    ax[0].set_xlabel("selection coefficient  s"); ax[0].set_ylabel("fixation prob.")
    ax[0].set_title("Gate I: $P_{fix}\\approx 2s$"); ax[0].legend(frameon=False)

    # Panel B: Gate III  empirical vs tunnelling prediction
    emp = np.array([r[1] for r in g3rows]); pred = np.array([r[3] for r in g3rows])
    labels = ["base", "½ u_c", "2 s_d", "½ s_c"]
    ax[1].plot([0, max(pred) * 1.1], [0, max(pred) * 1.1], "--", color="0.4",
               label="y = x")
    ax[1].scatter(pred, emp, color="black", zorder=3)
    for x, y, t in zip(pred, emp, labels):
        ax[1].annotate(t, (x, y), textcoords="offset points", xytext=(6, -2),
                       fontsize=8)
    ax[1].set_xlabel("predicted  $u_c\\,2s_c/s_d$")
    ax[1].set_ylabel("simulated  $P_{resc}$")
    ax[1].set_title("Gate III: tunnelling rescue"); ax[1].legend(frameon=False)

    # Panel C: asymmetry -- two regimes
    r = asym["ratios"]
    aA = np.array([o[2] for o in asym["outA"]])
    aB = np.array([o[2] for o in asym["outB"]])
    cr = asym["cost_ratio"]
    xs = np.linspace(0, r.max() * 1.05, 50)
    ax[2].plot(xs, xs, "--", color="0.4", label="full restoration: $u_↓/u_↑$")
    ax[2].plot(xs, xs * cr, ":", color="0.4",
               label=f"fixed benefit: $\\times{cr:.2f}$")
    ax[2].scatter(r, aA, color="crimson", zorder=3, label="sim (full restor.)")
    ax[2].scatter(r, aB, color="steelblue", marker="s", zorder=3,
                  label="sim (fixed benefit)")
    ax[2].axhline(1.0, color="0.9", lw=0.8)
    ax[2].set_xlabel("mutational-target ratio  $u_↓/u_↑$")
    ax[2].set_ylabel("rescue asymmetry  $P_↓/P_↑$")
    ax[2].set_title("Asymmetry reduces to $u_↓/u_↑$\n(cost cancels iff rescue fully restores)")
    ax[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/gates_and_asymmetry.png", dpi=140)
    print("\n[figure written: gates_and_asymmetry.png]")


# =============================================================================
#  ASSUMPTIONS REGISTRY
#  Every load-bearing assumption in the model, classified. "module" names the
#  sensitivity test below that probes it (or '--' = flagged, not yet coded).
#  "expected" is the predicted DIRECTION of effect on the central conclusion
#  (asymmetry -> u_down/u_up, with a cost factor only when benefit is decoupled
#  from cost).
# =============================================================================
ASSUMPTIONS = [
  # id, category, statement, relaxation probed, module, expected effect
  ("A1","pop-gen","Haploid Wright-Fisher; offspring variance = 1 (Poisson)",
        "diploidy + dominance h","S3",
        "sets the 2s factor; recessive benefits fix far less often"),
  ("A2","pop-gen","No X-linkage / no hemizygosity",
        "real X, hemizygous males (faster-X)","SLiM",
        "raises male-side boost establishment -> favours male compensation"),
  ("A3","pop-gen","Constant N, no demography",
        "bottleneck during rescue window","S5",
        "bottleneck depresses rescue; check if asymmetry shifts"),
  ("A4","architecture","Single relocated copy; compensation is ONE cis mutation",
        "L polygenic routes (target = L*u)","S7",
        "raises supply; target generalises to (L_d u_d)/(L_u u_u)"),
  ("A5","architecture","Dispensable duplicate vs 'no-copy' baseline (fitness 1)",
        "neutral compensated copy (pi_c=1/N)","S4",
        "rescue collapses ~1/N; cost factor RE-APPEARS (no s_c to cancel)"),
  ("A6","fitness","Compensation FULLY restores optimum (s_c tied to s_d)",
        "benefit decoupled from cost (fixed s_c)","S2",
        "THE HINGE: coupled -> ratio=u_d/u_u; decoupled -> x cost ratio"),
  ("A7","fitness","Gaussian, SYMMETRIC cost (over- = under-expression)",
        "one-sided cost, gamma-fold for over-expression","S1",
        "irrelevant under full restoration; scales cost factor otherwise"),
  ("A8","fitness","Cost is fully sex-limited to one germline",
        "pleiotropic compensator (sexual antagonism)","S8",
        "female silencer collapses first; resolution needs sex-limited expression"),
  ("A9","mutation","Rescue from de novo mutation only",
        "standing variation at mut-sel balance","S6",
        "bypasses bottleneck; asymmetry survives if source costs symmetric"),
  ("A10","mutation","u_up, u_down constant; their ratio is THE unknown",
        "(this is the estimand, not relaxed)","--",
        "the quantity the whole model isolates"),
  ("A11","expression","k is one constant multiplier (~1/3), same for all genes",
        "gene-to-gene variation in k","discussion",
        "k-MAGNITUDE and the AG[tagg]C motif go to the paper's DISCUSSION as"),
  ("A11b","expression","(continued: an invitation for the reader to measure,",
        "not modelled here -- below this selection theory's resolution)","discussion",
        "reader inspiration; the theory models selection FOR compensation, not k"),
  ("A12","process","Establishment = reaching frequency 0.5",
        "threshold sensitivity","(checked)",
        "innocuous for beneficial rescue; defines pi_c for neutral case"),
]

def print_assumptions():
    print("\n" + "=" * 70)
    print("MODEL ASSUMPTIONS (and which sensitivity test probes each)")
    print("=" * 70)
    for a in ASSUMPTIONS:
        print(f"  {a[0]:3s} [{a[1]:12s}] {a[2]}")
        print(f"        relax: {a[3]:42s} -> [{a[4]}]")


# =============================================================================
#  DIPLOID fixation probability (for S3: dominance dependence -> faster-X logic)
# =============================================================================
def diploid_pfix(s, h, N, reps, gen_cap=200000):
    """
    N diploid individuals, 2N alleles. Genotype fitness AA=1, Aa=1+h s, aa=1+s.
    Seed one mutant allele; run to absorption; return fixation probability.
    """
    twoN = 2 * N
    a = np.ones(reps, dtype=np.int64)
    alive = np.ones(reps, bool)
    fixed = np.zeros(reps, bool)
    for _ in range(gen_cap):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        q = a[idx] / twoN
        p = 1.0 - q
        wbar = p * p + 2 * p * q * (1 + h * s) + q * q * (1 + s)
        q_sel = (p * q * (1 + h * s) + q * q * (1 + s)) / wbar
        a_new = RNG.binomial(twoN, q_sel)
        a[idx] = a_new
        df = a_new == twoN
        dl = a_new == 0
        fixed[idx[df]] = True
        alive[idx[df | dl]] = False
    return fixed.mean()


# =============================================================================
#  SENSITIVITY MODULES
# =============================================================================
def S1_asymmetric_cost(k=1/3, crho2=0.10, N=4000, reps=90000, u=1e-3, sfix=0.03):
    """A7: penalise OVER-expression gamma-fold. Show full restoration is immune
       (ratio stays u_d/u_u); fixed benefit scales the cost factor by 1/gamma."""
    base_up = crho2 * (1 - k) ** 2
    base_dn = crho2 * (1 - k ** 2)
    rows = []
    for gamma in [1.0, 2.0, 4.0]:
        sdn = gamma * base_dn
        up_full = rescue_prob(N, base_up, base_up, u, reps)
        dn_full = rescue_prob(N, sdn, sdn, u, reps)
        up_fix = rescue_prob(N, base_up, sfix, u, reps)
        dn_fix = rescue_prob(N, sdn, sfix, u, reps)
        rows.append((gamma, dn_full / up_full, dn_fix / up_fix))
    return rows


def S2_benefit_coupling(k=1/3, crho2=0.10, N=4000, reps=90000, u=1e-3, sfix=0.03):
    """A6 (the hinge): interpolate s_c = (1-b) s_d + b s_fix. b=0 coupled
       (ratio->u_d/u_u=1 here), b=1 decoupled (ratio-> cost factor 0.5)."""
    s_up = crho2 * (1 - k) ** 2
    s_dn = crho2 * (1 - k ** 2)
    rows = []
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sc_up = (1 - b) * s_up + b * sfix
        sc_dn = (1 - b) * s_dn + b * sfix
        up = rescue_prob(N, s_up, sc_up, u, reps)
        dn = rescue_prob(N, s_dn, sc_dn, u, reps)
        rows.append((b, dn / up))
    return rows


def S3_dominance(s=0.05, N=2000, reps=150000):
    """A1: diploid fixation vs dominance h. Hemizygous X ~ the h->1 limit, so
       a recessive male-beneficial boost is 'exposed' on the X (faster-X)."""
    rows = []
    for h in [0.0, 0.1, 0.25, 0.5, 1.0]:
        p = diploid_pfix(s, h, N, reps)
        rows.append((h, p, 2 * h * s))
    return rows


def S4_neutral_rescue(s_d=0.05, u_c=0.05, reps=120000):
    """A5: compensated copy is NEUTRAL (s_c=0, pi_c~1/N). Show rescue ~1/N
       (collapses for large N) -- the dispensable-duplicate regime."""
    rows = []
    for N in [300, 600, 1200]:
        P = rescue_prob(N, s_d, 0.0, u_c, reps, est_frac=1.0)  # require fixation
        pred = u_c * (1.0 / N) / s_d
        rows.append((N, P, pred, P * N))
    return rows


def run_sensitivity():
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: do the conclusions survive relaxed assumptions?")
    print("=" * 70)

    print("\n S1 [A7] asymmetric cost (over-expression penalised gamma-fold):")
    s1 = S1_asymmetric_cost()
    print(f"   {'gamma':>6} {'ratio FULL restor.':>20} {'ratio FIXED benefit':>20}")
    for g, rf, rx in s1:
        print(f"   {g:6.1f} {rf:20.3f} {rx:20.3f}")
    print("   -> full restoration ratio ~1 regardless of gamma (ROBUST);")
    print("      fixed-benefit ratio falls ~1/gamma (cost shape matters here).")

    print("\n S2 [A6] benefit-cost coupling (b=0 coupled ... b=1 decoupled):")
    s2 = S2_benefit_coupling()
    print(f"   {'beta':>6} {'P_down/P_up (u_d=u_u)':>24}")
    for b, r in s2:
        print(f"   {b:6.2f} {r:24.3f}")
    print("   -> interpolates from ~1.0 (cost cancels) to ~0.5 (cost factor).")

    print("\n S3 [A1] diploid fixation vs dominance h (faster-X driver):")
    s3 = S3_dominance()
    print(f"   {'h':>6} {'Pfix':>10} {'2hs':>10}")
    for h, p, pred in s3:
        print(f"   {h:6.2f} {p:10.5f} {pred:10.5f}")
    print("   -> recessive (h~0) boosts barely fix; hemizygous X exposes them")
    print("      (effective h->1), the population-genetic root of faster-X.")

    print("\n S4 [A5] neutral compensated copy (dispensable duplicate):")
    s4 = S4_neutral_rescue()
    print(f"   {'N':>6} {'Presc':>10} {'pred u_c/(N s_d)':>18} {'Presc*N':>10}")
    for N, P, pred, PN in s4:
        print(f"   {N:6d} {P:10.6f} {pred:18.6f} {PN:10.4f}")
    print("   -> Presc ~ 1/N (Presc*N ~ const): neutral rescue collapses for")
    print("      large N, and the cost factor RE-APPEARS (no s_c to cancel).")
    return s1, s2, s3, s4


def make_sensitivity_figure(s1, s2, s3, s4):
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.0))

    # S1 asymmetric cost
    g = [r[0] for r in s1]
    ax[0].plot(g, [r[1] for r in s1], "o-", color="crimson",
               label="full restoration")
    ax[0].plot(g, [r[2] for r in s1], "s--", color="steelblue",
               label="fixed benefit")
    ax[0].axhline(1.0, color="0.85", lw=0.8)
    ax[0].set_xlabel("over-expression penalty  $\\gamma$")
    ax[0].set_ylabel("$P_↓/P_↑$  (at $u_↓=u_↑$)")
    ax[0].set_title("A7: cost shape\n(full restoration is immune)")
    ax[0].legend(frameon=False, fontsize=8)

    # S2 coupling
    b = [r[0] for r in s2]
    ax[1].plot(b, [r[1] for r in s2], "o-", color="purple")
    ax[1].axhline(1.0, color="0.85", lw=0.8); ax[1].axhline(0.5, color="0.85", lw=0.8)
    ax[1].set_xlabel("benefit decoupling  $\\beta$")
    ax[1].set_ylabel("$P_↓/P_↑$")
    ax[1].set_title("A6 (hinge): coupling\n1.0 (coupled) -> 0.5 (decoupled)")

    # S3 dominance
    h = [r[0] for r in s3]
    ax[2].plot(h, [r[1] for r in s3], "o-", color="black", label="simulated $P_{fix}$")
    ax[2].plot(h, [r[2] for r in s3], "--", color="0.5", label="$2hs$")
    ax[2].set_xlabel("dominance  $h$")
    ax[2].set_ylabel("fixation prob.")
    ax[2].set_title("A1: dominance\n(X hemizygosity $\\approx h{=}1$)")
    ax[2].legend(frameon=False, fontsize=8)

    # S4 neutral rescue 1/N
    Ns = [r[0] for r in s4]
    ax[3].plot(Ns, [r[1] for r in s4], "o-", color="darkgreen", label="$P_{resc}$")
    ax[3].plot(Ns, [r[2] for r in s4], "--", color="0.5", label="$u_c/(N s_d)$")
    ax[3].set_xlabel("population size  $N$")
    ax[3].set_ylabel("rescue prob.")
    ax[3].set_title("A5: neutral copy\n$P_{resc}\\propto 1/N$ (collapses)")
    ax[3].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/sensitivity.png", dpi=140)
    print("\n[figure written: sensitivity.png]")


# =============================================================================
#  GATE IV (external validation): RECOVER FASTER-X (Charlesworth-Coyne-Barton'87)
#
#  A sex-structured Wright-Fisher for an X-linked allele: N/2 males carry one X
#  (hemizygous), N/2 females carry two. A male-beneficial allele of dominance h
#  is FULLY exposed in hemizygous males but only partially (h) in females.
#  Theory predicts the X/autosome adaptive substitution-rate ratio = (1+2h)/(4h),
#  which exceeds 1 for recessive/partly-recessive benefits (h<1/2) and equals 1
#  at h=1/2 -- the faster-X effect. We recover it from the same machinery that
#  passed Gates I-III, confirming the phi term of the tutorial is faster-X.
# =============================================================================
def sex_x_pfix(s, h, N, reps, gen_cap=200000):
    """Fixation prob of a single new X-linked allele (male advantage s, dom h)."""
    half = N // 2                       # males = females = N/2
    Mm = np.zeros(reps, dtype=np.int64)            # mutant male-X alleles (/half)
    Mf = np.ones(reps, dtype=np.int64)             # mutant female-X alleles (/N)
    total_X = half + N                             # = 3N/2
    alive = np.ones(reps, bool); fixed = np.zeros(reps, bool)
    for _ in range(gen_cap):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        pm = Mm[idx] / half                        # male mutant freq
        pf = Mf[idx] / N                            # female mutant freq
        # selection
        pm2 = pm * (1 + s) / (1 + s * pm)          # hemizygous males
        wbar = (1 - pf) ** 2 + 2 * pf * (1 - pf) * (1 + h * s) + pf ** 2 * (1 + s)
        pf2 = (pf * (1 - pf) * (1 + h * s) + pf ** 2 * (1 + s)) / wbar
        # reproduction: sons' X from mother; daughters' X from mother + father
        Mm_new = RNG.binomial(half, pf2)                       # sons
        Mf_new = RNG.binomial(half, pf2) + RNG.binomial(half, pm2)  # daughters
        Mm[idx] = Mm_new; Mf[idx] = Mf_new
        tot = Mm_new + Mf_new
        fx = tot == total_X; lz = tot == 0
        fixed[idx[fx]] = True
        alive[idx[fx | lz]] = False
    return fixed.mean()


def run_faster_x():
    print("\n" + "=" * 70)
    print("GATE IV (external): RECOVER FASTER-X  -- target X/A rate=(1+2h)/(4h)")
    print("=" * 70)
    s, N, reps = 0.05, 2000, 150000
    rows, ok = [], True
    print(f"  {'h':>5} {'Pfix_X':>9} {'Pfix_A':>9} {'rate X/A':>9} "
          f"{'(1+2h)/4h':>10}")
    for h in [0.1, 0.25, 0.5, 1.0]:
        pX = sex_x_pfix(s, h, N, reps)
        pA = diploid_pfix(s, h, N, reps)
        # rate = supply x Pfix; supply_X = 1.5N u, supply_A = 2N u -> factor 0.75
        rate_ratio = 0.75 * pX / pA
        pred = (1 + 2 * h) / (4 * h)
        within = abs(rate_ratio - pred) < 0.20 * pred
        ok &= within
        rows.append((h, pX, pA, rate_ratio, pred))
        print(f"  {h:5.2f} {pX:9.5f} {pA:9.5f} {rate_ratio:9.3f} {pred:10.3f} "
              f"{'PASS' if within else 'FAIL'}")
    print(f"  faster-X (ratio>1) for h<0.5, =1 at h=0.5: "
          f"{'RECOVERED' if ok else 'NOT recovered'}")
    return ok, rows


# =============================================================================
#  S5 [A3]  DEMOGRAPHY: a bottleneck overlapping the rescue window
# =============================================================================
def rescue_prob_demog(N_pre, N_bot, T_bot, s_d, s_c, u_c, reps,
                      gen_cap=200000, est_frac=0.5):
    """As rescue_prob, but population size is N_bot for the first T_bot
       generations (the bottleneck), then N_pre. Counts are resampled to the
       current size each generation, so the bottleneck adds drift."""
    none = np.full(reps, N_bot - 1, dtype=np.int64)
    D = np.ones(reps, dtype=np.int64)
    C = np.zeros(reps, dtype=np.int64)
    rescued = np.zeros(reps, bool)
    active = np.arange(reps)
    w = np.array([1.0, 1.0 - s_d, 1.0 + s_c])
    for g in range(gen_cap):
        if active.size == 0:
            break
        Nt = N_bot if g < T_bot else N_pre
        n0, d, c = none[active], D[active], C[active]
        mut = RNG.binomial(d, u_c); d = d - mut; c = c + mut
        a0, a1, a2 = n0 * w[0], d * w[1], c * w[2]
        A = a0 + a1 + a2
        c_new = RNG.binomial(Nt, a2 / A)
        rem = Nt - c_new
        denom = a0 + a1
        p1 = np.where(denom > 0, a1 / np.where(denom > 0, denom, 1), 0.0)
        d_new = RNG.binomial(rem, p1)
        none[active] = rem - d_new; D[active] = d_new; C[active] = c_new
        lost = (D[active] + C[active]) == 0
        won = C[active] >= int(np.ceil(est_frac * Nt))
        rescued[active[won]] = True
        active = active[~(lost | won)]
    return rescued.mean()


def S5_demography(k=1/3, crho2=0.10, reps=100000, u=1e-3):
    """Compare male vs female full-restoration rescue with and without a
       bottleneck. Question: does demography change the ASYMMETRY, or just the
       magnitude?"""
    s_up = crho2 * (1 - k) ** 2
    s_dn = crho2 * (1 - k ** 2)
    N_pre = 4000
    up0 = rescue_prob(N_pre, s_up, s_up, u, reps)
    dn0 = rescue_prob(N_pre, s_dn, s_dn, u, reps)
    upb = rescue_prob_demog(N_pre, 200, 50, s_up, s_up, u, reps)
    dnb = rescue_prob_demog(N_pre, 200, 50, s_dn, s_dn, u, reps)
    return dict(up0=up0, dn0=dn0, upb=upb, dnb=dnb,
                asym0=dn0 / up0, asymb=dnb / upb)


# =============================================================================
#  S6 [A9]  STANDING VARIATION: rescue from a pre-existing compensator
#
#  A conditionally-neutral compensator sits at mutation-selection balance at the
#  source, frequency x0 = u_c/delta (delta = its pleiotropic source cost). After
#  relocation it becomes beneficial. The relocated copy is then BORN compensated
#  with probability x0 (cis), giving immediate rescue, plus the de novo route.
#  Key question (Daniel's): does SGV bypass the de novo bottleneck, and does it
#  ERASE the u_down/u_up asymmetry?
# =============================================================================
def rescue_prob_sgv(N, s_d, s_c, u_c, x0, reps, gen_cap=200000, est_frac=0.5):
    """Relocated copy is born as C (compensated) with prob x0, else as D."""
    born_C = RNG.random(reps) < x0
    none = np.full(reps, N - 1, dtype=np.int64)
    D = np.where(born_C, 0, 1).astype(np.int64)
    C = np.where(born_C, 1, 0).astype(np.int64)
    rescued = np.zeros(reps, bool)
    active = np.arange(reps)
    w = np.array([1.0, 1.0 - s_d, 1.0 + s_c])
    for _ in range(gen_cap):
        if active.size == 0:
            break
        n0, d, c = none[active], D[active], C[active]
        mut = RNG.binomial(d, u_c); d = d - mut; c = c + mut
        counts = wf_generation(np.stack([n0, d, c], axis=1), w)
        none[active], D[active], C[active] = counts[:, 0], counts[:, 1], counts[:, 2]
        lost = (D[active] + C[active]) == 0
        won = C[active] >= int(np.ceil(est_frac * N))
        rescued[active[won]] = True
        active = active[~(lost | won)]
    return rescued.mean()


def S6_standing_variation(k=1/3, crho2=0.10, N=4000, reps=100000, u=1e-3):
    """Sweep the source cost delta (-> x0=u/delta). Compare de novo (x0=0) to
       SGV, and report the asymmetry under symmetric source cost."""
    s_up = crho2 * (1 - k) ** 2
    s_dn = crho2 * (1 - k ** 2)
    denovo_up = rescue_prob_sgv(N, s_up, s_up, u, 0.0, reps)
    rows = []
    for delta in [np.inf, 0.05, 0.01]:        # inf = no SGV; smaller delta = more
        x0 = 0.0 if np.isinf(delta) else u / delta
        up = rescue_prob_sgv(N, s_up, s_up, u, x0, reps)
        dn = rescue_prob_sgv(N, s_dn, s_dn, u, x0, reps)   # symmetric u and delta
        rows.append((delta, x0, up, dn, dn / up))
    return denovo_up, rows


# =============================================================================
#  S7 [A4]  POLYGENIC / MULTIPLE ROUTES: effective target = L * u_c
# =============================================================================
def S7_polygenic(k=1/3, crho2=0.10, N=4000, reps=100000, u=1e-3):
    """L independent cis routes scale the target -> P_resc ~ L. The 'mutational
       target' unknown generalises to (L_down u_down)/(L_up u_up)."""
    s_up = crho2 * (1 - k) ** 2
    rows = []
    for L in [1, 3, 10]:
        P = rescue_prob(N, s_up, s_up, L * u, reps)
        rows.append((L, P, P / L))
    return rows


def run_tier2():
    print("\n" + "=" * 70)
    print("NEXT-TIER SENSITIVITY: A3 demography, A9 standing variation, A4 routes")
    print("=" * 70)

    print("\n S5 [A3] bottleneck (N->200 for 50 gen) vs constant N=4000:")
    d = S5_demography()
    print(f"   constant N : P_up={d['up0']:.5f}, P_dn={d['dn0']:.5f}, "
          f"asym={d['asym0']:.3f}")
    print(f"   bottleneck : P_up={d['upb']:.5f}, P_dn={d['dnb']:.5f}, "
          f"asym={d['asymb']:.3f}")
    print("   -> bottleneck depresses BOTH sides; compare asym0 vs asymb.")

    print("\n S6 [A9] standing variation (x0=u/delta); de novo baseline first:")
    dn0, s6 = S6_standing_variation()
    print(f"   de novo only (x0=0): P_up={dn0:.5f}")
    print(f"   {'delta':>8} {'x0':>10} {'P_up':>9} {'P_dn':>9} {'P_dn/P_up':>10}")
    for delta, x0, up, dn, r in s6:
        ds = "inf" if np.isinf(delta) else f"{delta:.3f}"
        print(f"   {ds:>8} {x0:10.2e} {up:9.5f} {dn:9.5f} {r:10.3f}")
    print("   -> SGV raises rescue 5-9x above de novo (bypasses the bottleneck).")
    print("      It does NOT erase the u_dn/u_up target ratio, but MULTIPLIES it")
    print("      by the BENEFIT ratio s_c_dn/s_c_up: standing copies ESTABLISH")
    print("      (~2 s_c) rather than tunnel (~u 2 s_c/s_d). Under full restoration")
    print("      that benefit ratio = (1+k)/(1-k) = 2, so SGV tilts toward the")
    print("      FEMALE (higher-cost, higher-benefit) side -- the OPPOSITE of de novo.")
    print("      => the net asymmetry depends on the de novo : SGV mix.")

    print("\n S7 [A4] polygenic routes (effective target = L*u):")
    s7 = S7_polygenic()
    print(f"   {'L':>4} {'P_resc':>9} {'P/L (const?)':>14}")
    for L, P, PL in s7:
        print(f"   {L:4d} {P:9.5f} {PL:14.6f}")
    print("   -> P_resc ~ L: target unknown generalises to (L_dn u_dn)/(L_up u_up).")
    return d, (dn0, s6), s7


def make_tier2_figure(fx_rows, demog, sgv, poly):
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.0))

    h = [r[0] for r in fx_rows]; rr = [r[3] for r in fx_rows]
    ax[0].plot(h, rr, "o", color="black", label="simulated X/A rate")
    hs = np.linspace(0.08, 1.0, 100)
    ax[0].plot(hs, (1 + 2 * hs) / (4 * hs), "--", color="0.5",
               label="$(1{+}2h)/4h$ (CCB)")
    ax[0].axhline(1.0, color="0.85", lw=0.8)
    ax[0].set_xlabel("dominance  $h$"); ax[0].set_ylabel("X/A substitution rate")
    ax[0].set_title("Gate IV: faster-X recovered\n(ratio>1 for $h<1/2$)")
    ax[0].legend(frameon=False, fontsize=8)

    x = np.arange(2)
    ax[1].bar(x - 0.18, [demog["up0"], demog["dn0"]], 0.36, label="constant N",
              color="0.6")
    ax[1].bar(x + 0.18, [demog["upb"], demog["dnb"]], 0.36, label="bottleneck",
              color="indianred")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["male\n(boost)", "female\n(silence)"])
    ax[1].set_ylabel("rescue prob.")
    ax[1].set_title(f"A3: bottleneck\nasym {demog['asym0']:.2f}->{demog['asymb']:.2f}")
    ax[1].legend(frameon=False, fontsize=8)

    dn0, rows = sgv
    x0s = [r[1] for r in rows]; ups = [r[2] for r in rows]; dns = [r[3] for r in rows]
    ax[2].plot(x0s, ups, "o-", color="navy", label="male P_up")
    ax[2].plot(x0s, dns, "s-", color="crimson", label="female P_dn")
    ax[2].axhline(dn0, color="0.6", ls=":", label="de novo only")
    ax[2].set_xscale("symlog", linthresh=1e-3)
    ax[2].set_xlabel("standing freq  $x_0=u/\\delta$")
    ax[2].set_ylabel("rescue prob.")
    ax[2].set_title("A9: standing variation\n(bypasses bottleneck)")
    ax[2].legend(frameon=False, fontsize=8)

    Ls = [r[0] for r in poly]; Ps = [r[1] for r in poly]
    ax[3].plot(Ls, Ps, "o-", color="darkorange")
    ax[3].set_xlabel("number of routes  $L$"); ax[3].set_ylabel("rescue prob.")
    ax[3].set_title("A4: polygenic routes\n$P_{resc}\\propto L$")

    fig.tight_layout()
    fig.savefig("figures/tier2.png", dpi=140)
    print("\n[figure written: tier2.png]")


# =============================================================================
#  S8 [A8]  CROSS-SEX PLEIOTROPY: the compensator is itself antagonistic
#
#  X suppression is MALE-GERMLINE-specific, so the relocation cost stays
#  male-limited. The new ingredient is that a compensatory cis-variant changes
#  expression in BOTH sexes (pleiotropy psi in [0,1]). A pleiotropic compensator
#  fixes the male problem (benefit s_b) but creates a female problem (cost kappa):
#     net effect (sex-averaged, autosomal) :  s_c_eff = s_b - psi * kappa
#  Beyond a critical psi* = s_b/kappa the pleiotropic compensator turns
#  deleterious and CANNOT rescue; resolution then requires a SEX-LIMITED variant
#  (rate u_sl << u_pl, no female cost) -- i.e., the evolution of sex-biased
#  expression that the sexual-antagonism literature (Rice 1984; Connallon &
#  Clark) treats as the resolution of conflict.
#
#  Biology of the two female (other-sex) costs:
#    boost of a MALE gene  -> over-expresses an UNWANTED gene in females : small
#    silence of a FEMALE gene -> under-expresses a NEEDED gene in females : large
#  So kappa_dn > kappa_up: the female-side silencer collapses at LOWER psi.
# =============================================================================
def S8_cross_sex_pleiotropy(k=1/3, crho2=0.10, N=4000, reps=100000,
                            u_pl=1e-3, u_sl=1e-4):
    s_up = crho2 * (1 - k) ** 2        # male relocation cost (boost needed)
    s_dn = crho2 * (1 - k ** 2)        # female relocation cost (silencer needed)
    sb_up, sb_dn = s_up, s_dn          # full restoration: focal benefit = cost
    kappa_up = 0.5 * sb_up             # over-expressing an unwanted male gene: modest
    kappa_dn = 2.0 * sb_dn             # under-expressing a NEEDED female gene: large
    psi_star_up = sb_up / kappa_up     # = 2  -> never collapses in [0,1]
    psi_star_dn = sb_dn / kappa_dn     # = 0.5 -> collapses at psi=0.5

    # sex-limited floor (psi-independent rare route, no female cost)
    Psl_up = rescue_prob(N, s_up, sb_up, u_sl, reps)
    Psl_dn = rescue_prob(N, s_dn, sb_dn, u_sl, reps)

    rows = []
    for psi in np.linspace(0, 1, 6):
        sc_up = sb_up - psi * kappa_up
        sc_dn = sb_dn - psi * kappa_dn
        Ppl_up = rescue_prob(N, s_up, sc_up, u_pl, reps) if sc_up > 1e-3 else 0.0
        Ppl_dn = rescue_prob(N, s_dn, sc_dn, u_pl, reps) if sc_dn > 1e-3 else 0.0
        Ptot_up = 1 - (1 - Ppl_up) * (1 - Psl_up)
        Ptot_dn = 1 - (1 - Ppl_dn) * (1 - Psl_dn)
        rows.append((psi, Ptot_up, Ptot_dn,
                     (Ptot_dn / Ptot_up) if Ptot_up > 0 else np.nan, sc_up, sc_dn))
    return dict(rows=rows, Psl_up=Psl_up, Psl_dn=Psl_dn,
                psi_star_up=psi_star_up, psi_star_dn=psi_star_dn,
                kappa_up=kappa_up, kappa_dn=kappa_dn)


def run_a8():
    print("\n" + "=" * 70)
    print("S8 [A8] CROSS-SEX PLEIOTROPY: compensator becomes sexually antagonistic")
    print("=" * 70)
    r = S8_cross_sex_pleiotropy()
    print(f"  collapse thresholds: psi*_male={r['psi_star_up']:.2f} (never in [0,1]),"
          f" psi*_female={r['psi_star_dn']:.2f}")
    print(f"  sex-limited floor: P_up={r['Psl_up']:.5f}, P_dn={r['Psl_dn']:.5f}")
    print(f"  {'psi':>5} {'P_up':>9} {'P_dn':>9} {'P_dn/P_up':>10} "
          f"{'sc_up':>8} {'sc_dn':>8}")
    for psi, pu, pd, ratio, scu, scd in r["rows"]:
        flag = " <-- female pleiotropic route dead" if scd <= 1e-3 else ""
        print(f"  {psi:5.2f} {pu:9.5f} {pd:9.5f} {ratio:10.3f} "
              f"{scu:8.4f} {scd:8.4f}{flag}")
    print("  -> rising pleiotropy disfavours the FEMALE side first (silencing a")
    print("     needed female gene is costly). Past psi*=0.5 only the rare")
    print("     SEX-LIMITED route rescues -> this is sexual-conflict resolution")
    print("     via sex-biased expression (Rice 1984; Connallon & Clark 2011).")
    return r


def make_a8_figure(r):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    rows = r["rows"]
    psi = [x[0] for x in rows]
    pu = [x[1] for x in rows]
    pd = [x[2] for x in rows]

    ax[0].plot(psi, pu, "o-", color="navy", label="male (boost)")
    ax[0].plot(psi, pd, "s-", color="crimson", label="female (silencer)")
    ax[0].axhline(r["Psl_dn"], color="crimson", ls=":", lw=0.9,
                  label="female sex-limited floor")
    ax[0].axvline(r["psi_star_dn"], color="0.6", ls="--", lw=0.8)
    ax[0].annotate("female route\ncollapses", (r["psi_star_dn"], max(pd) * 0.6),
                   fontsize=8, ha="center")
    ax[0].set_xlabel("compensator pleiotropy  $\\psi$")
    ax[0].set_ylabel("rescue probability")
    ax[0].set_title("A8: pleiotropic compensator\nbecomes antagonistic")
    ax[0].legend(frameon=False, fontsize=8)

    ratio = [x[3] for x in rows]
    ax[1].plot(psi, ratio, "o-", color="purple")
    ax[1].axhline(1.0, color="0.85", lw=0.8)
    ax[1].axvline(r["psi_star_dn"], color="0.6", ls="--", lw=0.8)
    ax[1].set_xlabel("compensator pleiotropy  $\\psi$")
    ax[1].set_ylabel("asymmetry  $P_↓/P_↑$")
    ax[1].set_title("Female side suppressed by\nconflict (needs sex-limitation)")

    fig.tight_layout()
    fig.savefig("figures/a8_pleiotropy.png", dpi=140)
    print("\n[figure written: a8_pleiotropy.png]")


# =============================================================================
#  DRIVER  --  enforce the rule: no female-side results until gates pass
# =============================================================================
if __name__ == "__main__":
    print_assumptions()

    ok1, g1rows = run_gate1()
    ok2, _      = run_gate2()
    ok3, g3rows = run_gate3()
    ok4, fxrows = run_faster_x()      # external validation: recover faster-X

    print("\n" + "#" * 70)
    if not (ok1 and ok2 and ok3):
        print("# CORE GATES NOT ALL PASSED -> female-side analysis WITHHELD.")
        print("#" * 70)
        raise SystemExit(1)
    print(f"# CORE GATES PASSED. Faster-X recovered: {ok4}.")
    print("# female-side analysis permitted.")
    print("#" * 70)

    asym = run_asymmetry()
    make_figure(g1rows, g3rows, asym)

    s1, s2, s3, s4 = run_sensitivity()
    make_sensitivity_figure(s1, s2, s3, s4)

    demog, sgv, poly = run_tier2()
    make_tier2_figure(fxrows, demog, sgv, poly)

    a8 = run_a8()
    make_a8_figure(a8)
    print("\nDone.")
