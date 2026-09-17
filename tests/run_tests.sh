#!/usr/bin/env bash
#
# Build and run the layer test suite.
#
#   ./tests/run_tests.sh              # plain build
#   ./tests/run_tests.sh --asan       # + AddressSanitizer (catches out-of-bounds
#                                     #   reads/writes that the plain build survives)
#   ./tests/run_tests.sh --simd       # + -fopenmp-simd, which makes the `#pragma omp
#                                     #   simd` directives in core/ actually apply
#   ./tests/run_tests.sh --asan Grad  # only tests whose name contains "Grad"
#
# Run it with --simd at least once before shipping. The pragmas are discarded in
# silence without the flag, so a malformed one is invisible until something turns
# them on -- which is exactly how the bogus array-section `reduction` clause in
# softmaxCrossEntropyDerivative survived: the suite passed without the flag and
# failed with it.
#
set -uo pipefail

cd "$(dirname "$0")"

CXX=${CXX:-clang++}
FLAGS=(-std=c++20 -g -O0 -Wall -Wextra -Wno-unknown-pragmas -Wno-ignored-qualifiers)
SUITES=(test_layer test_network)
SUFFIX=""
FILTER=""

for arg in "$@"; do
  case "$arg" in
    --asan)
      FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
      SUFFIX="${SUFFIX}_asan"
      ;;
    --simd)
      # Apple clang rejects -fopenmp but accepts -fopenmp-simd, which is all core/
      # needs: it uses `#pragma omp simd` and no parallel regions.
      FLAGS+=(-fopenmp-simd)
      SUFFIX="${SUFFIX}_simd"
      ;;
    -*)
      echo "unknown option: $arg" >&2; exit 2 ;;
    *)
      FILTER="$arg" ;;
  esac
done

mkdir -p build
echo "building with $CXX ${FLAGS[*]}"

# ASan's leak checker is not the point here and interferes with the forked
# children, so only the memory-error checks stay on.
export ASAN_OPTIONS="detect_leaks=0:abort_on_error=0"

status=0
for suite in "${SUITES[@]}"; do
  out="build/${suite}${SUFFIX}"
  "$CXX" "${FLAGS[@]}" "${suite}.cpp" -o "$out" || exit 1
  "./$out" "$FILTER" || status=1
done
exit $status
