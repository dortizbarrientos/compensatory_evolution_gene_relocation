"""Driver: validate the LRT null calibration, then map power vs (genes, effect)."""
import numpy as np, time
import phylo_silencer_power as P

RNG = np.random.default_rng(7)
N_TIPS, DEPTH = 12, 50.0          # Drosophila-scale: 12 species, ~50 My
edges, root, n_nodes, n_tips = P.make_tree(N_TIPS, DEPTH, RNG)

# baseline rates (per My)
A = B = 0.02                       # relocation X<->A (we study relocators)
C0 = 0.004                         # silencer gain when NOT needed (on X)
D = 0.02                           # silencer loss
ARGS = (edges, root, n_nodes, n_tips)


def indep_Q():
    return P.build_Q(A, A, B, B, C0, C0, D, D)

def dependent_Q(r):
    # model's prediction: silencer gain on autosome is r-fold faster
    return P.build_Q(A, A, B, B, C0, C0 * r, D, D)


def calibrate(n_rep=400, G=20):
    print("=" * 68)
    print(f"NULL CALIBRATION  (independent truth, G={G}, {n_rep} reps)")
    print("  the LRT must be ~5% false-positive and statistic ~chi^2(df)")
    print("=" * 68)
    Q = indep_Q()
    foc, fullp = [], []
    for _ in range(n_rep):
        ts = P.simulate(Q, *ARGS, G, RNG)
        s1, d1, p1 = P.lrt("indep", "focused", *ARGS, ts)
        foc.append((s1, p1))
    foc = np.array(foc)
    print(f"  focused (1 df): mean stat={foc[:,0].mean():.2f} (expect ~1.0); "
          f"FPR@0.05={np.mean(foc[:,1]<0.05):.3f} (expect ~0.05)")
    return foc


def power_curve(Glist, rlist, n_rep=200):
    print("\n" + "=" * 68)
    print(f"POWER  (focused 1-df test; {n_rep} reps/cell)")
    print("=" * 68)
    print(f"  {'effect r':>9} | " + " ".join(f"G={g:<4}" for g in Glist))
    table = {}
    for r in rlist:
        Q = dependent_Q(r)
        row = []
        for G in Glist:
            hits = 0
            for _ in range(n_rep):
                ts = P.simulate(Q, *ARGS, G, RNG)
                _, _, p = P.lrt("indep", "focused", *ARGS, ts)
                hits += (p < 0.05)
            row.append(hits / n_rep)
        table[r] = row
        print(f"  {r:9.0f} | " + " ".join(f"{v:5.2f} " for v in row))
    return table


if __name__ == "__main__":
    t0 = time.time()
    foc = calibrate(n_rep=400, G=20)
    table = power_curve(Glist=[5, 10, 20, 40], rlist=[5, 15, 40], n_rep=200)
    print(f"\n[elapsed {time.time()-t0:.0f}s]")
    np.savez("results/silencer_power.npz",
             foc=foc, Glist=[5,10,20,40], rlist=[5,15,40],
             table=np.array([table[r] for r in [5,15,40]]))
    print("saved silencer_power.npz")
