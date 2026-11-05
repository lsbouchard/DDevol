#!/usr/bin/env bash
# All numerical work below is performed by native executables.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build=${DDEVOL_BUILD:-"$root/build"}
out=${DDEVOL_RESULTS:-"$root/results"}
stage=${1:-help}
mkdir -p "$out"
run() { "$build/measure" "$out/$1" "$build/$2" "${@:3}"; }
core() {
  for name in convergence truncation growth sizes second pair moments; do
    run "$name" dense_benchmarks "$name"
  done
  run general general_tensor
  run complete ddevol_bench complete
  run ising ddevol_bench ising
  for dt in .08 .04 .02; do run "xy_$dt" ddevol_bench xy 1000 "$dt" 20; done
  run moments1000 ring_moments 1000
  run bounds moment_bounds "$out/moments1000.csv" bounds
  run hankel moment_bounds "$out/moments1000.csv" hankel
  run lanczos lanczos_reference 16
  run lanczos_small lanczos_reference 6
  run krylov_tpgp ddevol_bench ring 16 .04 1e-5 3.2
}
longtime() {
  for dt in .08 .04 .02; do
    run "ring_${dt}_1e-5" ddevol_bench ring 1000 "$dt" 1e-5 1.76 "$out/recovery_${dt}_1e-5.csv"
  done
  run ring_.04_3e-5 ddevol_bench ring 1000 .04 3e-5 2.8 "$out/recovery_.04_3e-5.csv"
  run ring_.04_1e-4 ddevol_bench ring 1000 .04 1e-4 3.2 "$out/recovery_.04_1e-4.csv"
  for dt in .04 .02 .01 .005; do run "short_${dt}_1e-12" ddevol_bench ring 1000 "$dt" 1e-12 .2; done
  for dt in .01 .005; do run "short_${dt}_1e-14" ddevol_bench ring 1000 "$dt" 1e-14 .2; done
}
finite() {
  for n in 8 12 16 24 1000; do
    run "finite_${n}_1e-4" ddevol_bench ring "$n" .04 1e-4 3.2
    run "finite_${n}_1e-5" ddevol_bench ring "$n" .04 1e-5 1.6
  done
  for n in 8 12 16 24 64; do run "finite_${n}_1e-6" ddevol_bench ring "$n" .04 1e-6 1.2; done
}
cubic() {
  for dt in .04 .02 .01; do run "cubic6_2_${dt}_1e-5" cubic 6 2 "$dt" 1e-5 .16 0; done
  for l in 4 8; do
    run "cubic${l}_2_.02_1e-5" cubic "$l" 2 .02 1e-5 .16 0
    run "cubic${l}_2_.02_1e-4_long" cubic "$l" 2 .02 1e-4 .5 0
  done
  run cubic6_2_.02_1e-4 cubic 6 2 .02 1e-4 .16 0
  run cubic6_1_.02_1e-5 cubic 6 1 .02 1e-5 .16 0
  run cubic6_shift cubic 6 2 .02 1e-5 .16 43
  run cubic6_shift_long cubic 6 2 .02 1e-4 .5 43
  for dt in .02 .01 .005; do run "cubic6_short_$dt" cubic 6 2 "$dt" 1e-7 .04 0; done
  run cubic6_tight_long cubic 6 2 .02 1e-5 .38 0
  run cubic6_coarse_long cubic 6 2 .04 1e-4 .48 0
  for shells in 1 2; do run "cubic6_${shells}_long" cubic 6 "$shells" .02 1e-4 1 0; done
  run lattice_m2 cubic shells
}
bounded() {
  # These independent runs use the original thresholds and the same output grid.
  export TPGP_ORBIT_BUCKETS=16
  for n in 16 1000; do
    steps=30; [[ $n != 1000 ]] || steps=32
    coarse=1e-7; fine=3e-8
    if [[ $n == 1000 ]]; then coarse=3e-8; fine=1e-8; fi
    TPGP_ORBIT_SAVE="$out/bounded${n}_coarse.ck" run "bounded${n}_coarse" orbit_fid_rev "$n" .03125 "$coarse" "$steps" 4000000 8000000
    TPGP_ORBIT_SAVE="$out/bounded${n}_fine.ck" run "bounded${n}_fine" orbit_fid_rev "$n" .015625 "$fine" "$((2*steps))" 4000000 8000000
    run "fid${n}_coarse" analyze orbit "$out/bounded${n}_coarse.csv"
    run "fid${n}_fine" analyze orbit "$out/bounded${n}_fine.csv"
  done
  run fulltrace16 sector_reference 16 60 .03125
}
short_bounds() {
  run bounded_short1000 tpgp_rev --model j1j2 --n 1000 --dt .0078125 --time .0078125 --threshold .00001 --cap 100000 --validated --residual-depth 0
  run bounded_short64 tpgp_rev --model j1j2 --n 64 --dt .0078125 --time .125 --threshold .00001 --cap 100000 --validated --residual-depth 1
  run independent_reference test_reference_rev
  run local_cluster check_cluster_rev
  run exact_small_moments check_moments_rev
}
cost() {
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  for n in 8 16 17; do
    run "cost${n}_spectral" sector_reference "$n" 30 .0625
    buckets=8; [[ $n != 8 ]] || buckets=1
    TPGP_ORBIT_BUCKETS=$buckets run "cost${n}_orbit" orbit_fid_rev "$n" .03125 1e-7 30 4000000 8000000
    run "cost${n}_fid" analyze orbit "$out/cost${n}_orbit.csv"
    run "cost${n}_difference" analyze compare "$out/cost${n}_fid.csv" F "$out/cost${n}_spectral.csv" F
  done
}
case "$stage" in
  core) core;; long) longtime;; finite) finite;; cubic) cubic;; bounded) bounded;; cost) cost;; short-bounds) short_bounds;;
  all) core; longtime; finite; cubic; short_bounds; bounded; cost;;
  *) printf '%s\n' 'Usage: bash scripts/campaign.sh core|long|finite|cubic|short-bounds|bounded|cost|all'; exit 2;;
esac
