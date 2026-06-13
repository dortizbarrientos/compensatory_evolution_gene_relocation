#!/usr/bin/env python3
# =====================================================================
#  run_multichrom.py   (SLiM 5; recapitating + parallel)
#  Driver + analysis for the X-vs-autosome linked-selection ladder.
#
#  WHAT CHANGED, AND WHY
#  ---------------------
#  Forward simulations do not reliably coalesce at every position, so we
#  now RECAPITATE each chromosome instead of demanding full forward
#  coalescence. Recapitation is done at the correct per-compartment
#  effective size:
#       autosome -> N
#       X        -> 0.75 N
#  For neutral runs this is exact (the deep past is genuinely neutral).
#  For selected runs the recapitated tail is neutral, so we report how
#  much of each genealogy was already settled forward (frac_fwd); a tail
#  near zero means the neutral correction is negligible.
#
#  The runs are independent across (cell, seed), so they are dispatched
#  to a process pool. SLiM itself is single-threaded here; we parallelise
#  at the replicate level, which is where the work is.
#
#  PIPELINE PER (cell, seed)
#  -------------------------
#  1. Run SLiM -> trees archive directory (one .trees per chromosome).
#  2. For each chromosome:
#       - load; if X, remove vacant (null) haplosomes;
#       - record frac_fwd = fraction of the sequence already coalesced
#         forward (diagnostic; computed BEFORE recapitation);
#       - recapitate at N (A) or 0.75 N (X);
#       - branch diversity (target 4N for A, 3N for X);
#       - neutral runs also paint mutations and report site pi.
#
#  REQUIREMENTS:  pip install -r requirements.txt ; SLiM 5 on PATH.
# =====================================================================

from __future__ import annotations

import os
# Keep BLAS single-threaded per worker so the process pool does not
# oversubscribe cores. Must be set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import subprocess
import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np
import tskit
import msprime
import pyslim


# ---------------------------------------------------------------------
#  Configuration  (sanity-check scale)
# ---------------------------------------------------------------------

SLIM = os.environ.get("SLIM_BINARY", "slim")
SLIM_SCRIPT = os.environ.get("SLIM_SCRIPT", "xa_multichrom.slim")
WORKDIR = os.environ.get("WORKDIR", "archives")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 4))

N_DIPLOID = 1000
L_BP = 1_000_000
R_PER_BP = 1e-8
MU_NEUTRAL = 1e-8
BURNIN = 10               # burn-in in units of N

MU_DEL = 1e-8
SEL_S = -0.02
SEL_H = 0.5

N_REPLICATES = 20

SYM_A, SYM_X = "A", "X"
VACANT_SYMBOLS = {"X", "Y", "MT"}

# Recapitation effective sizes, per compartment.
NE_RECAP = {SYM_A: N_DIPLOID, SYM_X: int(round(0.75 * N_DIPLOID))}


@dataclass(frozen=True)
class Cell:
    label: str
    regime: str            # "neutral" or "bgs"
    sel_s: float = SEL_S


MATRIX = [
    Cell("neutral", "neutral"),               # rungs 0 + 1
    Cell("bgs",     "bgs"),                    # rung 2
    Cell("bgs_s0",  "bgs", sel_s=0.0),         # control: s=0 must recover neutral
]


# ---------------------------------------------------------------------
#  SLiM run
# ---------------------------------------------------------------------

def run_slim(label: str, regime: str, sel_s: float, seed: int) -> str:
    os.makedirs(WORKDIR, exist_ok=True)
    outdir = os.path.join(WORKDIR, f"{label}_seed{seed}")
    args = [
        SLIM, "-s", str(seed),
        "-d", f"REGIME='{regime}'",
        "-d", f"N={N_DIPLOID}",
        "-d", f"L={L_BP}",
        "-d", f"R={R_PER_BP}",
        "-d", f"MU_DEL={MU_DEL}",
        "-d", f"SEL_S={sel_s}",
        "-d", f"SEL_H={SEL_H}",
        "-d", f"BURNIN={BURNIN}",
        "-d", f"OUTDIR='{outdir}'",
        SLIM_SCRIPT,
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SLiM failed for {label} seed {seed}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return outdir


def find_archive_files(outdir: str) -> dict:
    """Map chromosome symbol -> .trees path. Files are named
    chromosome_<symbol>.trees, so the symbol is the trailing token."""
    mapping = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*.trees"))):
        stem = os.path.basename(f).rsplit(".trees", 1)[0]
        mapping[stem.split("_")[-1]] = f
    return mapping


# ---------------------------------------------------------------------
#  Load, recapitate, measure
# ---------------------------------------------------------------------

def forward_coalesced_fraction(ts: tskit.TreeSequence) -> float:
    """Fraction of the sequence already coalesced (single root) by the end
    of the forward sim. Computed BEFORE recapitation. Near 1.0 means the
    neutral recapitated tail is negligible. For the X this must be computed
    AFTER remove_vacant, or the isolated null nodes inflate the root count.
    """
    coalesced = sum(t.span for t in ts.trees() if t.num_roots == 1)
    return coalesced / ts.sequence_length


def load_chrom(path: str, symbol: str, seed: int):
    """Load one chromosome, clean the X, recapitate at the right N_e.
    Returns (recapitated_ts, frac_fwd)."""
    ts = tskit.load(path)
    if symbol in VACANT_SYMBOLS:
        ts = pyslim.remove_vacant(ts)              # drop male null-X haplosomes

    frac_fwd = forward_coalesced_fraction(ts)      # diagnostic, pre-recapitation

    ts = pyslim.recapitate(
        ts,
        ancestral_Ne=NE_RECAP[symbol],             # N for A, 0.75N for X
        recombination_rate=R_PER_BP,
        random_seed=seed,
    )
    max_roots = max(t.num_roots for t in ts.trees())
    assert max_roots == 1, f"{path}: recapitation left {max_roots} roots"
    return ts, frac_fwd


def branch_diversity(ts: tskit.TreeSequence) -> float:
    """Mean pairwise branch length per site, in generations.
    Targets: 4N (autosome), 3N (X)."""
    return float(ts.diversity(mode="branch"))


def site_pi(ts: tskit.TreeSequence, seed: int) -> float:
    """Nucleotide diversity after neutral mutation overlay (confirmation).
    Target ~ 4*N*mu for the autosome."""
    next_id = pyslim.next_slim_mutation_id(ts)
    mts = msprime.sim_mutations(
        ts, rate=MU_NEUTRAL, random_seed=seed, keep=True,
        model=msprime.SLiMMutationModel(type=0, next_id=next_id),
    )
    return float(mts.diversity(mode="site"))


# ---------------------------------------------------------------------
#  One unit of work (runs in a worker process)
# ---------------------------------------------------------------------

def run_one_task(task):
    label, regime, sel_s, seed = task
    outdir = run_slim(label, regime, sel_s, seed)
    files = find_archive_files(outdir)
    tsA, fracA = load_chrom(files[SYM_A], SYM_A, seed)
    tsX, fracX = load_chrom(files[SYM_X], SYM_X, seed)
    bA, bX = branch_diversity(tsA), branch_diversity(tsX)
    piA = site_pi(tsA, seed) if regime == "neutral" else None
    return label, bA, bX, piA, min(fracA, fracX)


# ---------------------------------------------------------------------
#  Orchestrate and report
# ---------------------------------------------------------------------

def main() -> None:
    tasks = [(c.label, c.regime, c.sel_s, seed)
             for c in MATRIX for seed in range(1, N_REPLICATES + 1)]
    agg = {c.label: {"bA": [], "bX": [], "piA": [], "frac": []} for c in MATRIX}

    print(f"Running {len(tasks)} tasks on {MAX_WORKERS} workers ...")
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one_task, t): t for t in tasks}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            label, bA, bX, piA, frac = fut.result()
            d = agg[label]
            d["bA"].append(bA); d["bX"].append(bX); d["frac"].append(frac)
            if piA is not None:
                d["piA"].append(piA)
            print(f"  [{n}/{len(tasks)}] {label}")

    def stats(x):
        x = np.asarray(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(x.size)

    results = {}
    for label, d in agg.items():
        mA, _ = stats(d["bA"]); mX, _ = stats(d["bX"])
        ratio = (np.asarray(d["bX"]) / np.asarray(d["bA"]))
        results[label] = {
            "bA": mA, "bX": mX,
            "ratio": ratio.mean(), "ratio_se": ratio.std(ddof=1) / np.sqrt(ratio.size),
            "frac": np.mean(d["frac"]),
            "piA": (stats(d["piA"])[0] if d["piA"] else None),
        }

    tA, tX = 4 * N_DIPLOID, 3 * N_DIPLOID
    t_piA = 4 * N_DIPLOID * MU_NEUTRAL

    print("\n" + "=" * 90)
    print(f"{'cell':<10}{'branch A':>11}{'branch X':>11}{'X/A':>9}"
          f"{'A/4N':>9}{'X/3N':>9}{'frac_fwd':>11}")
    print("-" * 90)
    for label, r in results.items():
        print(f"{label:<10}{r['bA']:>11.1f}{r['bX']:>11.1f}{r['ratio']:>9.3f}"
              f"{r['bA']/tA:>9.3f}{r['bX']/tX:>9.3f}{r['frac']:>11.4f}")
    print("=" * 90)

    nt = results["neutral"]
    print(f"\nRUNG 0  branch A = {nt['bA']:.1f}   [target {tA}]")
    print(f"RUNG 1  X/A      = {nt['ratio']:.3f} +/- {nt['ratio_se']:.3f}   [target 0.750]")
    if nt["piA"] is not None:
        print(f"        site pi  = {nt['piA']:.3e}   [target {t_piA:.3e}]")

    B_A = results["bgs"]["bA"] / nt["bA"]
    B_X = results["bgs"]["bX"] / nt["bX"]
    B_A0 = results["bgs_s0"]["bA"] / nt["bA"]
    print(f"\nRUNG 2  B (A) = {B_A:.3f}   B' (X) = {B_X:.3f}"
          f"   [X more reduced: {'OK' if B_X < B_A else 'FAIL'}]")
    print(f"        control s=0: B = {B_A0:.3f}   "
          f"[-> 1: {'OK' if abs(B_A0 - 1) < 0.05 else 'CHECK'}]")
    print("\n  frac_fwd near 1.0 means the neutral recapitated tail is")
    print("  negligible. If a bgs cell shows a low frac_fwd, raise BURNIN.")


if __name__ == "__main__":
    main()
