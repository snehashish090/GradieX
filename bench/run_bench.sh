#!/usr/bin/env bash
#
# Benchmark v0.0.3 against v0.0.2 across optimisation levels and OpenMP settings.
#
#   ./bench/run_bench.sh              # full matrix, CSV on stdout
#   ./bench/run_bench.sh > out.csv
#
# Both engines are driven by the same harness (bench/bench.cpp), so the timing
# loop, the data and the architectures are identical by construction.
#
# On OpenMP: core/ uses `#pragma omp simd` and nothing else -- no parallel
# regions, no runtime calls. Apple clang rejects -fopenmp outright, but accepts
# -fopenmp-simd, which enables exactly the SIMD directives and needs no runtime
# library. That is the flag to compare against, not -fopenmp.

set -uo pipefail
cd "$(dirname "$0")/.."

CXX=${CXX:-clang++}
BUILD=${BUILD:-bench/build}
mkdir -p "$BUILD"

BASE=(-std=c++20 -Wno-unknown-pragmas -Wno-ignored-qualifiers)

echo "version,arch,params,flags,batch,median_us,min_us,max_us,iters"

for opt in -O2 -O3; do
  for omp in "" "-fopenmp-simd"; do
    label="${opt}${omp:+ ${omp}}"
    tag=$(echo "$label" | tr -d ' -' | tr '[:upper:]' '[:lower:]')

    # shellcheck disable=SC2086
    "$CXX" "${BASE[@]}" $opt $omp -DBENCH_V2 bench/bench.cpp -o "$BUILD/v2_$tag" || exit 1
    # shellcheck disable=SC2086
    "$CXX" "${BASE[@]}" $opt $omp            bench/bench.cpp -o "$BUILD/v3_$tag" || exit 1

    "$BUILD/v2_$tag" "$label" 1
    "$BUILD/v3_$tag" "$label" 1
    "$BUILD/v3_$tag" "$label" 32
  done
done
