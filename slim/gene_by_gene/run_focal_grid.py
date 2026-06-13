#!/usr/bin/env python3
# =====================================================================
#  run_focal_grid.py   (SLiM 5; BGS-first; parallel)
#  The relocation experiment over a parameter grid.
#
#  PER (grid point, seed):
#    1. Run focal_grid.slim -> trees archive + stdout with SUBCOUNT lines.
#    2. Parse substitution counts (one neutral type per chromosome).
#    3. For each chromosome: load, remove vacant (X), recapitate at the
#       compartment Ne (X -> 0.75N, A -> N), measure FOCAL-WINDOW branch
#       diversity (= 4 Ne B, in generations).
#    4. Substitution rate k = subs / (LFOCAL * interval_generations).
#
#  READOUTS
#    polymorphism  : focal branch diversity   (X -> 3N, Aq -> 4N, Al -> 4N*B)
#    relocation    : Al/X (target (4/3)*B_A), Aq/X (target 4/3), B_A = Al/Aq
#    Birky-Walsh   : k_X ~ k_Aq ~ k_Al ~ MU_FOCAL  (divergence ruler is clean)
#
#  IDENTITY: when k = mu, the HKA statistic pi/k equals the branch
#  diversity, so branch diversity IS the poly/div ratio. The substitution
#  rung confirms the denominator would be clean for an empiricist.
#
#  REQUIREMENTS: pip install -r requirements.txt ; SLiM 5 on PATH.
# =====================================================================

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
import subprocess
import concurrent.futures as cf

import numpy as np
import tskit
import msprime
import pyslim


# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------

SLIM = os.environ.get("SLIM_BINARY", "slim")
SLIM_SCRIPT = os.environ.get("SLIM_SCRIPT", "focal_grid.slim")
WORKDIR = os.environ.get("WORKDIR", "archives_focal")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 4))

N_DIPLOID = 1000
L_BP = 2_000_000          # each chromosome
LFOCAL = 200_000          # central focal window
MU_FOCAL = 1e-8
BURNIN = 10
N_REPLICATES = 12

# Map each chromosome symbol to its recapitation Ne and its focal
# neutral mutation-type id (must match focal_grid.slim: m2/m3/m4).
SYMBOLS = ["X", "Aq", "Al"]
NE_RECAP = {"X": int(round(0.75 * N_DIPLOID)), "Aq": N_DIPLOID, "Al": N_DIPLOID}
MTID = {2: "X", 3: "Aq", 4: "Al"}

# ---- The grid (BGS first). Each point overrides the defaults. --------
# 'sel_s' = 0.0 is the CONTROL: no load, so B should be 1 everywhere.
DEFAULTS = dict(sel_s=-0.02, sel_h=0.5, mu_del=1e-8, R=1e-8)
GRID = [
    dict(label="control_s0",  sel_s=0.0),                   # B -> 1 (apparatus check)
    dict(label="bgs_r1e-8",   R=1e-8),                      # tight linkage
    dict(label="bgs_r1e-7",   R=1e-7),                      # looser linkage; B closer to 1
    dict(label="bgs_load2x",  mu_del=2e-8),                 # heavier load; smaller B
]


def params(point: dict) -> dict:
    p = dict(DEFAULTS); p.update(point); return p


# ---------------------------------------------------------------------
#  Analytic background-selection prediction (Nordborg-Charlesworth-
#  Charlesworth 1996, leading order in small t and r)
# ---------------------------------------------------------------------
#  At a focal neutral position x, the diversity reduction from deleterious
#  sites in the flanks is  B(x) = exp(-E(x)), with
#       E(x) = INT u * t / (t + r(z))^2  dz   over the flanks,
#  where u = MU_DEL (per-bp haploid rate), t = sh (heterozygous effect),
#  and r(z) = R * |x - z| (linear map distance, fine at these scales).
#  The flank integrals are closed-form; we average B(x) across the focal
#  window because edge positions sit closer to the load than the centre.
#
#  Validity: deterministic BGS assumes selection is strong relative to
#  drift (Ne*t >> 1). Here Ne*t = N * sh; check it is comfortably > 1, or
#  the prediction degrades (interference, weak-selection effects).

def bgs_B_analytic(mu_del: float, sel_s: float, sel_h: float, R: float,
                   L: int, lfocal: int, n_x: int = 201) -> float:
    t = sel_h * abs(sel_s)                    # heterozygous effect
    if t == 0.0 or mu_del == 0.0:
        return 1.0
    fs = (L - lfocal) // 2
    fe = fs + lfocal - 1
    xs = np.linspace(fs, fe, n_x)
    # closed-form flank integrals (left flank [0, fs), right flank (fe, L])
    left = 1.0 / (t + R * (xs - fs)) - 1.0 / (t + R * xs)
    right = 1.0 / (t + R * (fe - xs)) - 1.0 / (t + R * (L - xs))
    E = (mu_del * t / R) * (left + right)
    return float(np.exp(-E).mean())


# ---------------------------------------------------------------------
#  SLiM run
# ---------------------------------------------------------------------

def run_slim(point: dict, seed: int):
    os.makedirs(WORKDIR, exist_ok=True)
    p = params(point)
    outdir = os.path.join(WORKDIR, f"{point['label']}_seed{seed}")
    args = [
        SLIM, "-s", str(seed),
        "-d", f"N={N_DIPLOID}",
        "-d", f"L={L_BP}",
        "-d", f"LFOCAL={LFOCAL}",
        "-d", f"R={p['R']}",
        "-d", f"MU_FOCAL={MU_FOCAL}",
        "-d", f"MU_DEL={p['mu_del']}",
        "-d", f"SEL_S={p['sel_s']}",
        "-d", f"SEL_H={p['sel_h']}",
        "-d", f"BURNIN={BURNIN}",
        "-d", f"OUTDIR='{outdir}'",
        SLIM_SCRIPT,
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SLiM failed for {point['label']} seed {seed}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return outdir, proc.stdout


def parse_stdout(text: str):
    """Pull substitution counts (per type id) and the measurement interval
    out of SLiM's stdout."""
    counts, interval = {}, None
    for line in text.splitlines():
        if line.startswith("SUBCOUNT,"):
            _, mtid, n = line.split(",")
            counts[int(mtid)] = int(n)
        elif line.startswith("INTERVAL,"):
            interval = int(line.split(",")[1])
    return counts, interval


# ---------------------------------------------------------------------
#  Trees -> focal-window branch diversity
# ---------------------------------------------------------------------

def find_archive_files(outdir: str) -> dict:
    mapping = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*.trees"))):
        stem = os.path.basename(f).rsplit(".trees", 1)[0]
        mapping[stem.split("_")[-1]] = f
    return mapping


def focal_branch_diversity(path: str, symbol: str, seed: int, recomb_rate: float) -> float:
    """Recapitate at the compartment Ne, then measure branch diversity in
    the focal window only. recomb_rate must match the run's R."""
    ts = tskit.load(path)
    if symbol == "X":
        ts = pyslim.remove_vacant(ts)
    ts = pyslim.recapitate(
        ts, ancestral_Ne=NE_RECAP[symbol],
        recombination_rate=recomb_rate, random_seed=seed,
    )
    # focal window = [FS, FE+1) ; windows must start at 0 and end at L
    fs = (L_BP - LFOCAL) // 2
    fe1 = fs + LFOCAL
    vals = ts.diversity(mode="branch", windows=[0, fs, fe1, ts.sequence_length])
    return float(vals[1])


# ---------------------------------------------------------------------
#  One unit of work
# ---------------------------------------------------------------------

def run_one_task(task):
    point, seed = task
    p = params(point)
    outdir, stdout = run_slim(point, seed)
    counts, interval = parse_stdout(stdout)
    files = find_archive_files(outdir)

    bdiv, krate = {}, {}
    for sym in SYMBOLS:
        bdiv[sym] = focal_branch_diversity(files[sym], sym, seed, p["R"])
    for mtid, sym in MTID.items():
        krate[sym] = counts.get(mtid, 0) / (LFOCAL * interval)
    return point["label"], bdiv, krate


# ---------------------------------------------------------------------
#  Orchestrate and report
# ---------------------------------------------------------------------

def main() -> None:
    tasks = [(pt, seed) for pt in GRID for seed in range(1, N_REPLICATES + 1)]
    agg = {pt["label"]: {"bX": [], "bAq": [], "bAl": [],
                         "kX": [], "kAq": [], "kAl": []} for pt in GRID}

    print(f"Running {len(tasks)} tasks on {MAX_WORKERS} workers ...")
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one_task, t): t for t in tasks}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            label, bdiv, krate = fut.result()
            d = agg[label]
            d["bX"].append(bdiv["X"]); d["bAq"].append(bdiv["Aq"]); d["bAl"].append(bdiv["Al"])
            d["kX"].append(krate["X"]); d["kAq"].append(krate["Aq"]); d["kAl"].append(krate["Al"])
            print(f"  [{n}/{len(tasks)}] {label}")

    def m(x): return float(np.mean(x))

    tN = N_DIPLOID
    point_by_label = {pt["label"]: pt for pt in GRID}

    # ---- Table 1: polymorphism + linked-selection prediction --------
    print("\n" + "=" * 96)
    print("POLYMORPHISM (focal branch diversity, generations) and BGS prediction")
    print(f"{'point':<13}{'X':>8}{'Aq':>8}{'Al':>8}{'Aq/X':>8}{'Al/X':>8}"
          f"{'B_sim':>8}{'B_pred':>8}{'sim/pred':>10}")
    print("-" * 96)
    for label, d in agg.items():
        bX, bAq, bAl = m(d["bX"]), m(d["bAq"]), m(d["bAl"])
        B_sim = bAl / bAq
        p = params(point_by_label[label])
        B_pred = bgs_B_analytic(p["mu_del"], p["sel_s"], p["sel_h"], p["R"], L_BP, LFOCAL)
        print(f"{label:<13}{bX:>8.0f}{bAq:>8.0f}{bAl:>8.0f}"
              f"{bAq/bX:>8.3f}{bAl/bX:>8.3f}{B_sim:>8.3f}{B_pred:>8.3f}"
              f"{B_sim/B_pred:>10.3f}")
    print("=" * 96)
    print(f"Neutral targets:  X ~ {3*tN}   Aq ~ {4*tN}   Aq/X ~ 1.333")
    print(f"B_sim = Al/Aq (linked-selection reduction);  B_pred = NCC96;  "
          f"sim/pred ~ 1 validates the theory.")
    print(f"Ne*t for the BGS points = {N_DIPLOID * DEFAULTS['sel_h'] * abs(DEFAULTS['sel_s']):.0f} "
          f"(want >> 1 for deterministic BGS to hold).")

    # ---- Table 2: Birky-Walsh (substitution rate must stay ~ mu) ----
    print("\n" + "=" * 60)
    print("BIRKY-WALSH: substitution rate must be flat at MU_FOCAL")
    print(f"{'point':<13}{'kX':>12}{'kAq':>12}{'kAl':>12}")
    print("-" * 60)
    for label, d in agg.items():
        print(f"{label:<13}{m(d['kX']):>12.2e}{m(d['kAq']):>12.2e}{m(d['kAl']):>12.2e}")
    print("-" * 60)
    print(f"target MU_FOCAL = {MU_FOCAL:.2e} everywhere (if k tracks the BGS")
    print("environment, the divergence ruler is broken and we stop).")
    print("=" * 60)


if __name__ == "__main__":
    main()
