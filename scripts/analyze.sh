#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build=${DDEVOL_BUILD:-"$root/build"}
out=${DDEVOL_RESULTS:-"$root/results"}
run() { "$build/measure" "$out/$1" "$build/analyze" "${@:2}"; }
run threshold_envelope envelope "$out/ring_.04_1e-4.csv" "$out/ring_.04_3e-5.csv" "$out/ring_.04_1e-5.csv" fid_average
run timestep_refinement refinement "$out/ring_.08_1e-5.csv" "$out/ring_.04_1e-5.csv" "$out/ring_.02_1e-5.csv" fid_average
run xy_refinement refinement "$out/xy_.08.csv" "$out/xy_.04.csv" "$out/xy_.02.csv" TPGP_correlation
run krylov_difference compare "$out/krylov_tpgp.csv" fid_average "$out/lanczos.csv" krylov_mean
run krylov_coverage containment "$out/krylov_tpgp.csv" fid_average "$out/lanczos.csv" lower95 upper95
for dt in .04 .02 .01 .005; do
  run "short_${dt}_membership" containment "$out/short_${dt}_1e-12.csv" fid_average "$out/bounds.csv" lower_M10 upper_M12
done
for n in 8 12 16 24; do
  for eta in 1e-4 1e-5; do
    run "finite_${n}_${eta}_difference" compare "$out/finite_${n}_${eta}.csv" fid_average "$out/finite_1000_${eta}.csv" fid_average
  done
  run "finite_${n}_1e-6_difference" compare "$out/finite_${n}_1e-6.csv" fid_average "$out/finite_64_1e-6.csv" fid_average
done
for file in "$out"/ring_*.csv "$out"/finite_*_1e-4.csv; do
  case "$file" in *.resources.csv) continue;; esac
  name=$(basename "$file" .csv)
  run "${name}_features" features "$file" fid_average
done
for name in cubic6_coarse_long cubic6_tight_long cubic6_shift_long cubic4_2_.02_1e-4_long cubic8_2_.02_1e-4_long cubic6_1_long; do
  run "${name}_difference" compare "$out/$name.csv" fid "$out/cubic6_2_long.csv" fid
done
run cubic_short_refinement refinement "$out/cubic6_short_.02.csv" "$out/cubic6_short_.01.csv" "$out/cubic6_short_.005.csv" fid
for n in 16 1000; do
  run "fid${n}_refinement" compare "$out/fid${n}_coarse.csv" F "$out/fid${n}_fine.csv" F
done
run fulltrace16_difference compare "$out/fid16_fine.csv" F "$out/fulltrace16.csv" F
