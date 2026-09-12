#!/usr/bin/env bash
#
# Build and run the layer test suite.
#
#   ./tests/run_tests.sh              # plain build
#   ./tests/run_tests.sh --asan       # + AddressSanitizer (catches out-of-bounds
#                                     #   reads/writes that the plain build survives)
#   ./tests/run_tests.sh --asan Grad  # only tests whose name contains "Grad"
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
      SUFFIX="_asan"
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
