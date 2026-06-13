"""
silencer_scoring.py
=====================================================================
The S-SCORING PIPELINE: turn male-germline expression into a CALIBRATED
silencer call S for the correlated-evolution test, and break the
active-vs-passive confound that would otherwise wreck it.

THE PROBLEM
  A relocated gene that sits BELOW the expression its genomic context
  predicts has been compensated -- but two very different things produce
  that negative residual:
    * ACTIVE silencer : a derived repressive element / piRNA / H3K9me3
                        pulls expression down  (the model's claim)
    * PASSIVE loss    : the retrocopy never carried its enhancers
                        (expression low for a boring reason)
  Both compensate; only the first is "evolving a silencer." Expression
  alone cannot tell them apart -- a molecular signal must.

THE PIPELINE (three steps, each validated below)
  1. EXPECTATION   regress male-germline expression on genomic context
                   (X is k-suppressed) + covariates; residual = obs - exp.
  2. MIXTURE       a 2-component model on residuals returns P(down-shifted)
                   = P(compensation happened), calibrated.
  3. MOLECULAR     combine P(down) with the molecular signal M to return
                   P(active silencer) -- the axis that separates active
                   from passive. Output S (soft, 0..1) for the phylo test.

DISCIPLINE: validate on simulated data where truth is known. Show (a) the
posteriors are calibrated, (b) residual alone CANNOT separate active from
passive (AUC ~ 0.5), and (c) the molecular axis restores that separation.
=====================================================================
"""
import numpy as np

RNG = np.random.default_rng(20260611)
LOG_K = np.log(1/3)          # X male-germline suppression, k~1/3 (Landeen 2016)


# ---------------------------------------------------------------------
# 0.  SIMULATE expression with known truth (normal / active / passive)
# ---------------------------------------------------------------------
def simulate(n=1500, p_active=0.10, p_passive=0.10, delta=1.5,
             mu_A=3.0, sigma=0.6, beta_len=0.4, rng=RNG):
    """Each gene: context C (0=X,1=A), latent class, log-expression, covariate,
    and a molecular signal M (active silencers carry it; passive loss does not)."""
    C = rng.integers(0, 2, size=n)                  # 0=X, 1=A
    length = rng.normal(0, 1, size=n)               # standardised covariate
    u = rng.random(n)
    klass = np.where(u < p_active, 1,               # 1 = active silencer
             np.where(u < p_active + p_passive, 2,  # 2 = passive loss
                      0))                            # 0 = normal
    expected = mu_A + (1 - C) * LOG_K + beta_len * length   # context + covariate
    shift = np.where(klass == 0, 0.0, -delta)       # silenced classes shifted down
    logE = expected + shift + rng.normal(0, sigma, size=n)
    # molecular signal: present for active silencers, rare otherwise
    M = np.where(klass == 1, rng.random(n) < 0.85,
                 rng.random(n) < 0.08).astype(int)
    return dict(C=C, length=length, logE=logE, klass=klass, M=M)


# ---------------------------------------------------------------------
# 1.  EXPECTATION model: residual = observed - context-predicted
# ---------------------------------------------------------------------
def expectation_residual(logE, C, length):
    X = np.column_stack([np.ones_like(logE), (1 - C), length])   # intercept, X-flag, cov
    beta, *_ = np.linalg.lstsq(X, logE, rcond=None)
    resid = logE - X @ beta
    return resid, beta


# ---------------------------------------------------------------------
# 2.  MIXTURE: 2-component 1-D Gaussian EM -> P(down-shifted)
# ---------------------------------------------------------------------
def gaussian_mixture_1d(x, iters=200, tol=1e-7):
    # init: "normal" near 0, "down" below
    mu = np.array([0.0, np.percentile(x, 15)])
    sd = np.array([x.std(), x.std()])
    w = np.array([0.8, 0.2])
    def npdf(v, m, s): return np.exp(-0.5*((v-m)/s)**2)/(s*np.sqrt(2*np.pi))
    ll_old = -np.inf
    for _ in range(iters):
        r0 = w[0]*npdf(x, mu[0], sd[0])
        r1 = w[1]*npdf(x, mu[1], sd[1])
        tot = r0 + r1 + 1e-300
        g1 = r1 / tot                                  # P(down | x)
        g0 = 1 - g1
        for j, g in enumerate((g0, g1)):
            Nk = g.sum()
            mu[j] = (g*x).sum()/Nk
            sd[j] = np.sqrt((g*(x-mu[j])**2).sum()/Nk + 1e-6)
            w[j] = Nk/len(x)
        # keep component 1 as the LOWER-mean (down) one
        if mu[0] < mu[1]:
            mu, sd, w = mu[::-1].copy(), sd[::-1].copy(), w[::-1].copy()
        ll = np.log(tot).sum()
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    r1 = w[1]*npdf(x, mu[1], sd[1])
    p_down = r1 / (w[0]*npdf(x, mu[0], sd[0]) + r1 + 1e-300)
    return p_down, dict(mu=mu, sd=sd, w=w)


# ---------------------------------------------------------------------
# 3.  MOLECULAR axis: P(active silencer) = P(down) * P(active | M)
#     A down-shifted gene WITH a molecular signal is an active silencer;
#     without it, it is more likely passive loss.
# ---------------------------------------------------------------------
def active_posterior(p_down, M, lr_present=6.0, lr_absent=0.25):
    # likelihood ratio for "active vs passive" given M, folded onto P(down)
    lr = np.where(M == 1, lr_present, lr_absent)
    odds_active = (p_down / (1 - p_down + 1e-9)) * lr
    return odds_active / (1 + odds_active)


# ---------------------------------------------------------------------
#  VALIDATION helpers
# ---------------------------------------------------------------------
def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = scores.argsort().argsort()
    return (ranks[labels == 1].mean() - (len(pos)-1)/2) / len(neg)


def calibration(p, truth, bins=8):
    edges = np.linspace(0, 1, bins+1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 10:
            rows.append((0.5*(lo+hi), truth[m].mean(), m.sum()))
    return np.array(rows)


# ---------------------------------------------------------------------
#  VALIDATION + FIGURE (run as __main__)
# ---------------------------------------------------------------------
def run_validation(n=3000, path="figures/silencer_scoring.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = simulate(n=n)
    resid, beta = expectation_residual(d["logE"], d["C"], d["length"])
    p_down, _ = gaussian_mixture_1d(resid)
    p_active = active_posterior(p_down, d["M"])
    kl = d["klass"]; mask = kl > 0
    active_true = (kl == 1).astype(int)
    down_true = (kl > 0).astype(int)

    auc_down = auc(p_down, down_true)
    auc_res  = auc(p_down[mask], active_true[mask])     # residual alone: active vs passive
    auc_mol  = auc(p_active[mask], active_true[mask])   # + molecular axis
    print(f"expectation X-suppression = {beta[1]:.2f} (truth {LOG_K:.2f})")
    print(f"AUC down-shift vs normal           = {auc_down:.3f}")
    print(f"AUC active vs passive (residual)   = {auc_res:.3f}  (expect ~0.5)")
    print(f"AUC active vs passive (+molecular) = {auc_mol:.3f}")
    cal = calibration(p_active, active_true)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for c, lab, col in [(0, "normal", "#888"), (1, "active silencer", "crimson"),
                        (2, "passive loss", "steelblue")]:
        ax[0].hist(resid[kl == c], bins=40, density=True, alpha=0.55, color=col, label=lab)
    ax[0].set_xlabel("expression residual (obs - context expectation)")
    ax[0].set_ylabel("density")
    ax[0].set_title("Active & passive overlap completely\nin the residual")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar([0, 1], [auc_res, auc_mol], color=["#bbb", "crimson"])
    ax[1].axhline(0.5, color="0.5", ls="--", lw=0.8); ax[1].set_ylim(0, 1)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["residual\nonly", "+ molecular\naxis"])
    ax[1].set_ylabel("AUC: active vs passive")
    ax[1].set_title("Only the molecular axis separates\nactive silencing from passive loss")
    for i, v in enumerate([auc_res, auc_mol]):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax[2].plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
    ax[2].plot(cal[:, 0], cal[:, 1], "o-", color="crimson")
    ax[2].set_xlabel("predicted P(active silencer)"); ax[2].set_ylabel("observed fraction")
    ax[2].set_title("Calibration: well-ranked,\nmildly over-confident")
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    run_validation()
