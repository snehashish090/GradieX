#!/usr/bin/env bash
#
# Benchmark v0.0.1, v0.0.2 and v0.0.3 with one harness.
#
#   ./bench/run_bench_all.sh                 # full matrix, CSV on stdout
#   ./bench/run_bench_all.sh --quick         # short runs, to smoke the harness
#   ./bench/run_bench_all.sh > all.csv
#
# All three engines are driven by bench/bench.cpp, so the timing loop, the data
# and the architectures are identical by construction. Each version is selected by
# a compile-time define:
#
#   -DBENCH_V1   0.0.1  -- compiled in through bench/v1/v001_shim.h
#   -DBENCH_V2   0.0.2  -- ../../0.0.2/core/network.h
#   (none)       0.0.3  -- ../core/network.h
#
# v0.0.2 is only measured at batch 1: it has no mini-batching, and its harness
# branch ignores the batch argument rather than pretending otherwise.
#
# On OpenMP: v0.0.3's core/ uses `#pragma omp simd` and nothing else -- no parallel
# regions, no runtime calls. Apple clang rejects -fopenmp outright but accepts
# -fopenmp-simd, which enables exactly those directives and needs no runtime
# library. v0.0.1 and v0.0.2 contain no pragmas, so the flag is expected to do
# nothing for them; they are still built with it so the comparison is like for like.

set -uo pipefail
cd "$(dirname "$0")/.."

CXX=${CXX:-clang++}
BUILD=${BUILD:-bench/build}
mkdir -p "$BUILD"

BASE=(-std=c++20 -Wno-unknown-pragmas -Wno-ignored-qualifiers)

QUICK=()
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=(--quick)
fi

echo "version,arch,params,flags,batch,median_us,min_us,max_us,iters,loss_first,loss_last"

for opt in -O2 -O3; do
  for omp in "" "-fopenmp-simd"; do
    label="${opt}${omp:+ ${omp}}"
    tag=$(echo "$label" | tr -d ' -' | tr '[:upper:]' '[:lower:]')

    # shellcheck disable=SC2086
    "$CXX" "${BASE[@]}" $opt $omp -DBENCH_V1 bench/bench.cpp -o "$BUILD/v1_$tag" || exit 1
    # shellcheck disable=SC2086
    "$CXX" "${BASE[@]}" $opt $omp -DBENCH_V2 bench/bench.cpp -o "$BUILD/v2_$tag" || exit 1
    # shellcheck disable=SC2086
    "$CXX" "${BASE[@]}" $opt $omp            bench/bench.cpp -o "$BUILD/v3_$tag" || exit 1

    # v0.0.1 supports mini-batching, so it gets the batch-32 column too.
    "$BUILD/v1_$tag" "$label" 1  "${QUICK[@]+"${QUICK[@]}"}"
    "$BUILD/v1_$tag" "$label" 32 "${QUICK[@]+"${QUICK[@]}"}"

    "$BUILD/v2_$tag" "$label" 1  "${QUICK[@]+"${QUICK[@]}"}"

    "$BUILD/v3_$tag" "$label" 1  "${QUICK[@]+"${QUICK[@]}"}"
    "$BUILD/v3_$tag" "$label" 32 "${QUICK[@]+"${QUICK[@]}"}"
  done
done
