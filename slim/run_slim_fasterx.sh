#!/usr/bin/env bash
# run_slim_fasterx.sh -- run the faster-X validation in SLiM.
# Requires a `slim` binary on PATH (SLiM >= 4; tested on 5.2).
# Build SLiM from https://github.com/MesserLab/SLiM if you don't have it:
#   git clone https://github.com/MesserLab/SLiM && cd SLiM && mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release .. && make slim -j4
#
# Writes results/slim_fasterx_results.txt (RESULT rows: chrom h s N gen_cap n_fixed Pfix).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if ! command -v slim >/dev/null 2>&1; then
  echo "ERROR: 'slim' not found on PATH. See header of this script to build it." >&2
  exit 1
fi

echo "Running SLiM faster-X validation (this can take several minutes)..."
slim slim/fasterx_validate.slim | grep '^RESULT' > results/slim_fasterx_results.txt
echo "Wrote results/slim_fasterx_results.txt:"
cat results/slim_fasterx_results.txt
