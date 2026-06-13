# X vs autosome — linked-selection validation ladder (SLiM 5)

Recover the standard neutral results first; creative work comes only after
the base is trusted.

## Files

| File | Role |
|---|---|
| `xa_multichrom.slim` | One sexual population, two chromosomes (autosome + X). Records a per-chromosome trees archive. |
| `run_multichrom.py` | Parallel driver: runs the matrix across seeds, recapitates each chromosome at the right N_e, computes branch and site diversity, prints results vs theory. |
| `requirements.txt` | Pinned Python deps (pyslim ≥ 1.1 is required for `remove_vacant`). |

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # SLiM 5 itself: separate binary on PATH
python run_multichrom.py                  # set MAX_WORKERS=8 to cap cores
```

## Recapitation (the corrected workflow)

Forward sims do not reliably coalesce at every position (`max_roots > 1`),
so we recapitate — but at the **right per-compartment effective size**:

- autosome → `ancestral_Ne = N`
- X → `ancestral_Ne = 0.75 N`

For neutral runs this is exact. For selected runs the recapitated tail is
neutral, so the driver reports `frac_fwd` (fraction of each genealogy
already settled by the forward sim). `frac_fwd ≈ 1` means the neutral tail
is negligible; if a `bgs` cell shows a low `frac_fwd`, raise `BURNIN`.

## What each rung must return

| Rung | Source | Headline (branch diversity, generations) | Pass |
|---|---|---|---|
| 0 | neutral, autosome | ≈ **4N** (= 4000); site π ≈ 4Nμ = 4e-5 | within ~2 SE |
| 1 | neutral, X | X/A ratio | → **0.750** |
| 2a | bgs, autosome | < 4N by *B* | *B* < 1; s=0 control → 1 |
| 2b | bgs, X | < 3N by *B′* | **B′ < B** |

## Parallelism

Runs are independent across (cell × seed) and dispatched to a process
pool (`MAX_WORKERS`, default = all cores). SLiM is single-threaded here, so
we parallelise at the replicate level. BLAS is pinned to one thread per
worker to avoid oversubscription.

## Caveat

The pipeline was assembled without a local SLiM, so the X path
(`remove_vacant` → recapitate at 0.75 N) gets its first real exercise once
rung 0 passes. If the X/A ratio drifts from 0.750, that step is where to
look.

## Next steps (in order)

1. Confirm rungs 0–1: A→4N, X/A→0.750, site π→4e-5.
2. Confirm rung 2 structural checks; sweep R and U; fit *B* to NCC96 (1996).
3. Add recurrent beneficials / moving optimum (draft); test whether the X
   depression grows with N.
4. Build the HKA triad (X/relocated locus, autosomal homolog, outgroup).
