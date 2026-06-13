#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- reproduce the analysis from scratch.
#
# Default (fast, ~minutes): install deps, run the Python analyses that
# regenerate every figure, and rebuild the figures that read committed
# results. Then compile the document.
#
# Expensive steps are GATED behind environment variables (off by default):
#   RUN_SLIM=1   build/run the SLiM faster-X validation (needs `slim`; ~min)
#   RUN_POWER=1  re-run the full correlated-evolution power grid (~30 min)
#
# Usage:
#   ./run_all.sh                 # fast path: figures + doc from committed results
#   RUN_SLIM=1 ./run_all.sh      # also regenerate slim_fasterx_results.txt
#   RUN_POWER=1 ./run_all.sh     # also regenerate pow_r*.npy from scratch
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"             # repo root
mkdir -p figures results

echo "==> [1/5] Installing Python dependencies"
pip install -r requirements.txt --quiet

echo "==> [2/5] Core analyses (regenerate figures + print validation)"
python src/wf_compensation.py        # gates I-IV + sensitivity -> 4 figures
python src/drosophila_forward.py     # calibrated forecast       -> drosophila_forward.png
python src/retrogene_data_test.py    # poly-vs-fixed data test    (prints stats)
python src/retrogene_data_test.py    # second call writes its figure
python src/silencer_scoring.py       # scoring pipeline + figure  -> silencer_scoring.png

echo "==> [3/5] (optional) SLiM faster-X validation"
if [ "${RUN_SLIM:-0}" = "1" ]; then
  bash slim/run_slim_fasterx.sh
else
  echo "    skipped (set RUN_SLIM=1 to run; using committed results/slim_fasterx_results.txt)"
fi

echo "==> [3b] (optional) full correlated-evolution power grid"
if [ "${RUN_POWER:-0}" = "1" ]; then
  python src/run_silencer_power.py     # ~30 min; overwrites results/*.npy/.npz
else
  echo "    skipped (set RUN_POWER=1 to run; using committed results/pow_r*.npy)"
fi

echo "==> [4/5] Figures that read committed results"
python src/make_fasterx_figure.py        # -> figures/fasterx_slim.png
python src/make_silencer_power_figure.py  # -> figures/silencer_power.png

echo "==> [5/5] Compile the document"
if command -v pdflatex >/dev/null 2>&1; then
  cd docs
  for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error model.tex >/dev/null 2>&1 || true; done
  cd ..
  echo "    docs/model.pdf rebuilt"
else
  echo "    pdflatex not found; skipping document build"
fi

echo "==> DONE. Figures in figures/, document in docs/model.pdf"
