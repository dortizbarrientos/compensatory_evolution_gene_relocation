"""
phylo_silencer_power.py
=====================================================================
A CORRELATED-EVOLUTION test for the compensatory SILENCER, plus the
power analysis that tells us -- before we touch real data -- whether the
handful of against-the-grain relocations on the Drosophila tree can
detect it.

THE TWO CHARACTERS (binary, evolving on a species phylogeny)
  C  context : 0 = X chromosome (male-germline suppressed, k~1/3)
               1 = autosome      (permissive; a low-optimum gene is now
                                  over-expressed and "wants" a silencer)
               -> observed per species; changes by RELOCATION.
  S  silencer: 0 = absent  (expression at context-default level)
               1 = present (derived repressive change pulling male-germline
                            expression back down)
               -> inferred from a NEGATIVE expression residual + molecular
                  signature (cis-element gain / piRNA / H3K9me3).

THE PREDICTION
  Silencer GAIN, S:0->1, should be ELEVATED when C=1 (on the autosome),
  because that is where the low-optimum gene is over-expressed. On the X
  the suppression is free, so no silencer is favoured. This is a coupling
  between a transition rate of one character and the STATE of the other --
  exactly Pagel's (1994) correlated-evolution hypothesis.

WHAT WE TEST
  M0  independent : C and S evolve independently            (4 rates)
  M1  focused     : M0 + silencer-gain depends on context   (5 rates) -> 1 df
                    (the model's specific directional prediction)
  M2  full Pagel  : all 8 rates free                         (8 rates) -> 4 df
  LRT: 2*dlnL ~ chi^2(df). We POOL across genes (shared rates, summed
  log-likelihoods) so the rare events accumulate into power.

DISCIPLINE: validate the NULL first. Under independent simulation the
focused LRT must be ~5% false-positive and the statistic ~chi^2(1); the
full Pagel LRT ~chi^2(4). Only then do we trust the power numbers.

State index s = 2*C + S :  0=(X,off) 1=(X,on) 2=(A,off) 3=(A,on)
=====================================================================
"""
import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import minimize
from scipy.stats import chi2

RNG = np.random.default_rng(20260611)


# ---------------------------------------------------------------------
# 1.  TREE  (ultrametric coalescent tree; parents get higher ids than kids)
# ---------------------------------------------------------------------
def make_tree(n_tips, depth, rng):
    active = list(range(n_tips))
    node_time = {i: 0.0 for i in active}
    edges = []                                   # (child, parent, length)
    nid = n_tips
    t = 0.0
    while len(active) > 1:
        k = len(active)
        t += rng.exponential(1.0 / (k * (k - 1) / 2.0))
        i, j = rng.choice(active, size=2, replace=False)
        for c in (i, j):
            edges.append((c, nid, t - node_time[c]))
        node_time[nid] = t
        active.remove(i); active.remove(j); active.append(nid)
        nid += 1
    root = active[0]
    scale = depth / t
    edges = [(c, p, L * scale) for (c, p, L) in edges]
    return edges, root, nid, n_tips


# ---------------------------------------------------------------------
# 2.  RATE MATRICES  (single-character changes only; duals forbidden)
#     params: a=C:0->1, b=C:1->0  (per S-state),  c=S:0->1, d=S:1->0 (per C-state)
# ---------------------------------------------------------------------
def build_Q(a0, a1, b0, b1, c0, c1, d0, d1):
    # rows/cols: 0=(X,off)1=(X,on)2=(A,off)3=(A,on)
    Q = np.zeros((4, 4))
    # C flips (X<->A): rate depends on S state (a*,b*)
    Q[0, 2] = a0; Q[1, 3] = a1                 # X->A  (S=0, S=1)
    Q[2, 0] = b0; Q[3, 1] = b1                 # A->X
    # S flips (off<->on): rate depends on C state (c*,d*)
    Q[0, 1] = c0; Q[2, 3] = c1                 # gain silencer (on X, on A)
    Q[1, 0] = d0; Q[3, 2] = d1                 # lose silencer (on X, on A)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def Q_from_params(p, model):
    a, b = p[0], p[1]                          # relocation rates (shared across S)
    if model == "indep":          # c,d shared across C
        c, d = p[2], p[3]
        return build_Q(a, a, b, b, c, c, d, d)
    if model == "focused":        # silencer GAIN depends on C: c0 (X), c1 (A)
        c0, c1, d = p[2], p[3], p[4]
        return build_Q(a, a, b, b, c0, c1, d, d)
    if model == "full":           # all 8 free
        return build_Q(*p)


NPARAM = {"indep": 4, "focused": 5, "full": 8}


# ---------------------------------------------------------------------
# 3.  PRUNING LIKELIHOOD  (vectorised over G genes; eigen-based expm)
# ---------------------------------------------------------------------
def pt_factory(Q):
    lam, V = eig(Q)
    Vi = inv(V)
    def Pt(t):
        return np.real((V * np.exp(lam * t)) @ Vi)
    return Pt


def loglik(Q, edges, root, n_nodes, n_tips, tip_states):
    # tip_states: (G, n_tips) int in {0..3}
    G = tip_states.shape[0]
    Pt = pt_factory(Q)
    partial = np.ones((n_nodes, G, 4))
    for tip in range(n_tips):
        partial[tip] = 0.0
        partial[tip, np.arange(G), tip_states[:, tip]] = 1.0
    children = {}
    for (c, p, L) in edges:
        children.setdefault(p, []).append((c, L))
    for node in range(n_tips, n_nodes):        # ascending id == post-order
        acc = np.ones((G, 4))
        for (c, L) in children[node]:
            acc *= np.einsum("ij,gj->gi", Pt(L), partial[c])
        partial[node] = acc
    pi = np.full(4, 0.25)                       # flat root prior
    L = partial[root] @ pi                      # (G,)
    return np.log(np.clip(L, 1e-300, None)).sum()


# ---------------------------------------------------------------------
# 4.  ML FIT  (optimise log-rates; a couple of restarts for stability)
# ---------------------------------------------------------------------
def fit(model, edges, root, n_nodes, n_tips, tip_states, restarts=3):
    k = NPARAM[model]
    best = (None, -np.inf)
    for r in range(restarts):
        x0 = np.log(RNG.uniform(0.005, 0.05, size=k))
        def nll(x):
            return -loglik(Q_from_params(np.exp(x), model),
                           edges, root, n_nodes, n_tips, tip_states)
        try:
            res = minimize(nll, x0, method="Nelder-Mead",
                           options=dict(maxiter=800, xatol=1e-3, fatol=1e-3))
            if -res.fun > best[1]:
                best = (np.exp(res.x), -res.fun)
        except Exception:
            pass
    return best


def lrt(modelA, modelB, *args):
    """LRT of nested modelA (simpler) inside modelB; returns (stat, df, p)."""
    _, llA = fit(modelA, *args)
    _, llB = fit(modelB, *args)
    stat = 2 * (llB - llA)
    df = NPARAM[modelB] - NPARAM[modelA]
    return max(stat, 0.0), df, chi2.sf(max(stat, 0.0), df)


# ---------------------------------------------------------------------
# 5.  SIMULATE tip states under a given Q (root from flat prior)
# ---------------------------------------------------------------------
def simulate(Q, edges, root, n_nodes, n_tips, G, rng):
    Pt = pt_factory(Q)
    state = np.full((n_nodes, G), -1, dtype=int)
    state[root] = rng.integers(0, 4, size=G)
    children = {}
    for (c, p, L) in edges:
        children.setdefault(p, []).append((c, L))
    for node in range(n_nodes - 1, n_tips - 1, -1):   # root..first internal
        if node not in children:
            continue
        for (c, L) in children[node]:
            P = np.clip(Pt(L), 0, None)
            P /= P.sum(axis=1, keepdims=True)
            cdf = np.cumsum(P[state[node]], axis=1)    # (G,4)
            u = rng.random((G, 1))
            state[c] = (u > cdf).sum(axis=1)
    return state[:n_tips].T                            # (G, n_tips)
