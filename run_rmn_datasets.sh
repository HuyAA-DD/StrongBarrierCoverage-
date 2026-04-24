#!/usr/bin/env bash
set -euo pipefail

run_if_needed () {
  local algo="$1"      # moead hoặc nsga
  local dataset="$2"   # ví dụ 100_3
  local done="results/${algo}/${dataset}.done"

  mkdir -p "results/${algo}"

  if [[ -f "$done" ]]; then
    echo "[SKIP] $algo $dataset (done flag exists: $done)"
    return 0
  fi

  echo "[RUN ] $algo $dataset"
  python "${algo}_exp.py" "$dataset"

  # nếu python chạy OK (exit code 0) thì mới mark done
  touch "$done"
  echo "[DONE] $algo $dataset"
}

for size in 100 150 200 250; do
  for i in {0..9}; do
    dataset="${size}_${i}"
    run_if_needed "moead" "$dataset"
    run_if_needed "nsga"  "$dataset"
  done
done