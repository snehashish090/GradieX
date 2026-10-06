# Contributing to GradieX

Thanks for your interest. GradieX is a small, dependency-free C++ neural network
library, and the goal is to keep it readable enough that someone can learn
backpropagation by reading the source.

## Getting started

```bash
git clone https://github.com/snehashish090/GradieX.git
cd GradieX

# build and run the C++ example
clang++ -std=c++20 -O2 -fopenmp-simd main.cpp -o main && ./main

# build and run both C++ suites
./tests/run_tests.sh
```

The engine is C++20 — `core/` uses `<numbers>` and `#pragma omp simd`. The runner
compiles `tests/test_layer.cpp` and `tests/test_network.cpp`, and a non-zero exit
means something failed, so it works directly in CI or a pre-commit hook.

To exercise the Python bindings as well:

```bash
python3 setup.py build_ext --inplace
MPLBACKEND=Agg python3 python/test_bindings.py
MPLBACKEND=Agg python3 python/test_plotting.py
```

`python/test_*.py` are assertion scripts rather than pytest tests: they import
`gradiex` from the release root and print `ok` per check. `MPLBACKEND=Agg` keeps the
plotting checks headless.

There is no build system on purpose. `core/` is header-only, `main.cpp` is one
translation unit, and `python/gradiex_module.cpp` is the only other one — it
`#include`s `core/network.h` and is compiled by `setup.py`.

## Before you open a pull request

1. **The suites pass.** `./tests/run_tests.sh` (91 checks across the two binaries) and
   the Python checks above — not just the ones near your change.
2. **Zero compiler warnings** under `-Wall -Wextra`.
3. **Clean under sanitizers**, which catches the out-of-bounds accesses a plain build
   survives:
   ```bash
   ./tests/run_tests.sh --asan
   ```
4. **Run `--simd` at least once.** The `#pragma omp simd` directives are discarded in
   silence when the build does not enable them, so a malformed clause is invisible
   until something turns them on:
   ```bash
   ./tests/run_tests.sh --simd
   ```
5. **New behaviour comes with a test.** If it can be wrong, it needs a check.
6. **Update the docs.** `README.md` documents the API; if you change it, change that,
   and add a `CHANGELOG.md` entry. If you change what the binding exposes, update
   `python/gradiex_module.cpp` too — the two are maintained by hand.

## Touching the backward pass

Anything in `core/layer.h` or `core/network.h` that touches gradients deserves care:
this is the code that can be wrong while still producing a network that trains
plausibly. The tests that decide whether you got it right are
`ComputeGradients_*`, `AccumulateGradients_*`, the `OutputLayer_ErrorSignal_*` /
`HiddenLayer_ErrorSignal_*` pairs, and
`Integration_ForwardBackwardUpdateLowersLoss`.

There is no longer a central-difference gradient check — v0.0.3's references are
computed in the tests from the definitions, per-case, rather than numerically. If you
add an activation, a loss, or a new path through `accumulateGradients`, add the case
that pins it down; the full list of what is and is not covered is in the
`tests/README.md` table.

## Invariants

The README maintains the authoritative list under
[Invariants](README.md#invariants) — flat-buffer offsets, the accumulator tree that
keeps the loop bit-exact, no allocation between `build()` and destruction, gradients
accumulating until the update step, and `input_cache_` holding the current pass's
input. Read it before changing the memory layout or the training loop. Breaking one
produces a library that trains plausibly and is silently wrong.

## Code style

Match the surrounding code rather than any external guide:

- Four-space indent, no tabs. Opening brace on the same line.
- `PascalCase` for types, `snake_case` for functions, variables, and members.
- **Spell things out.** This codebase uses `gradient_wrt_output`, `neuron_index`,
  `pre_activation` — not `g`, `i`, `z`. Loop counters included. The verbosity is
  deliberate; it is what makes the math readable next to the code.
- Comments explain *why*, not *what*. The math is in the file header; inline
  comments are for decisions a reader would otherwise question.
- No new dependencies. Standard library only. The Python layer adds `numpy` and
  `matplotlib` and nothing else.

## Good first contributions

The roadmap in `README.md` lists the open work. In rough order of value:

- **Momentum / Adam optimizers.** Plain SGD is the current ceiling on hard problems.
- **Model save/load.** The flat weight layout makes this close to two `memcpy`s plus
  a small header.
- **Early stopping driven by the loss history.**
- **Per-layer widths from the constructor.** The engine handles non-uniform networks;
  only the constructor is restrictive.
- **A trainable multi-class path.** Softmax is fused with cross-entropy and refuses to
  be evaluated on its own; restoring multi-class means the layer-wide path.

Smaller but genuinely useful: more worked examples, a CMake option for people who
want one, or feature normalization helpers (the engine assumes roughly unit-scale
inputs and does not scale for you).

## Reporting bugs

Open an issue with the compiler and version, the OS, a minimal `main()` that
reproduces it, and what you expected instead. If it is a numerical problem, include
the seed — every run is reproducible from one.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
