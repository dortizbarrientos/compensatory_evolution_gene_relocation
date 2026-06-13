# Compensatory regulatory evolution after gene relocation between the X and an autosome

A population-genetic model of what happens when a gene relocates between the
suppressed X and a permissive autosome in the *Drosophila* male germline, asking
whether **male-origin** relocations (which need a regulatory *boost*) and
**female-origin** relocations (which need a *silencer*) compensate differently —
and whether the data can see it.

The full write-up is `docs/model.pdf` (three parts: the analytical model, its
computational validation, and the confrontation with data + the silencer-test
design). This repository holds the code, figures, results, and data behind it.

## Quick start

```bash
./run_all.sh          # install deps, regenerate all figures, rebuild the document (~minutes)
```

Expensive steps are off by default and gated behind env vars:

```bash
RUN_SLIM=1  ./run_all.sh    # also rebuild the SLiM faster-X result (needs a `slim` binary)
RUN_POWER=1 ./run_all.sh    # also re-run the full power grid (~30 min)
```

## Layout

```
.
├── run_all.sh            reproducibility driver (fast path + gated heavy steps)
├── requirements.txt      Python deps (numpy, scipy, matplotlib, xlrd, python-docx)
├── src/                  analysis code
│   ├── wf_compensation.py            Wright-Fisher engine: Gates I-IV + sensitivity A1-A12
│   ├── drosophila_forward.py         Drosophila-calibrated forward forecast
│   ├── retrogene_data_test.py        poly-vs-fixed retrogene data test (two species)
│   ├── phylo_silencer_power.py       correlated-evolution test machinery (4-state CTMC, Pagel)
│   ├── run_silencer_power.py         null calibration + power grid driver (slow)
│   ├── silencer_scoring.py           expression-residual silencer-scoring pipeline
│   ├── make_fasterx_figure.py        rebuild faster-X figure from committed SLiM results
│   └── make_silencer_power_figure.py rebuild power-curve figure from committed arrays
├── slim/                 SLiM models
│   ├── fasterx_validate.slim         faster-X validation on a real X (TESTED)
│   ├── compensation_faster_x.slim    rescue-on-X skeleton (Gate III; WIP)
│   └── run_slim_fasterx.sh           build/run helper
├── data/
│   ├── raw/              published supplementary tables (Schrider 2011/2013) — see data/README.md
│   └── README.md         provenance + what's still needed for the silencer test
├── results/             numeric outputs (slim_fasterx_results.txt, pow_r*.npy, silencer_power.npz)
├── figures/             generated PNGs (9)
└── docs/                model.tex/pdf (main write-up) + earlier records (review, transplant)
```

## What each piece establishes

| Component | Result |
|---|---|
| `wf_compensation.py` | Gates I–IV pass: fixation $P\approx 2s$; recurrent establishment; tunnelling $P_{resc}\approx u\,2s_c/s_d$; faster-X $(1+2h)/4h$. Sensitivity A1–A12, five-ratio synthesis. |
| `fasterx_validate.slim` | Faster-X recovered on a real X in a second engine (crossover at $h=0.5$). |
| `drosophila_forward.py` | Four-cell survival spans $\sim10^4$; out-of-X excess is a net flux ($f_{male}$ knob); female-origin X→A floor near zero across regimes. |
| `retrogene_data_test.py` | X-origin retrogenes enriched among fixed vs polymorphic in **two species** (neutrality index 0.17–0.31); positive selection on the escape, mechanism supplied by the model. |
| `phylo_silencer_power.py` + `run_silencer_power.py` | Correlated-evolution test; LRT calibrated (FPR 0.04); power reaches 80–99% at the predicted strong coupling with ~8–15 against-grain genes. |
| `silencer_scoring.py` | Calibrated silencer call; residual alone can't separate active from passive silencing (AUC 0.51), the molecular axis restores it (0.74). |

## Notes
- Figures in `figures/` are committed (pre-generated); `run_all.sh` regenerates them.
- The two heavy steps (SLiM build/run, full power grid) are gated; their committed
  outputs in `results/` let every figure rebuild in minutes without them.
- Known pre-submission to-dos (non-blocking): verify a few bibliography page numbers;
  the SLiM faster-X run used $N=500$, $s=0.1$ for tractability.
- **Next:** assemble the against-the-grain gene list (sex-biased expression +
  fixed-relocation polarizations) to place a real $G$ on the power curve — see the
  data-gathering protocol.
