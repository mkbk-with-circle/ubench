#!/usr/bin/env bash
set -euo pipefail

echo "benchmark,device,blocks,size_bytes,adjusted_slope_us,adjusted_intercept_us,adjusted_r2,raw_slope_us,raw_intercept_us,raw_r2"

for csv in "$@"; do
  [[ -f "${csv}" ]] || continue
  awk -F, '
    NR == 2 {
      print $1 "," $2 "," $3 "," $4 "," $12 "," $13 "," $14 "," $9 "," $10 "," $11
    }
  ' "${csv}"
done
