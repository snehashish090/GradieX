# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on version ordering.** `0.0.2` was an architecture rewrite that temporarily
> dropped features present in `0.0.1`, and `0.0.3` continues that line. The numbering
> is non-monotonic in feature count by design: a lower count at a higher number marks
> the architecture reset, not a mistake. `0.0.3` has since regained mini-batching,
> shuffling, seeded initialisation and a multi-class path.

## [0.0.3] - 2026-09-17

Single precision, SIMD pragmas, and mini-batching. The Python bindings and the test
suite were ported to match.

### Added

- **Mini-batch training.** `Network(numHidden, numOutput, learningRate, batches)`.
  Gradients accumulate over a batch, are scaled by `1/batchSize`, then applied once.
  `batches = 0` means one batch per epoch.
- **Per-epoch shuffling**, seeded and optional, inside `train()`. It lives in the
  engine so every caller gets the same batching and the same update rule.
- **Seeded weight initialisation** — `initializers::setSeed(unsigned)`, exposed as
  `gradiex.seed(n)`. Weight init was previously unreproducible.
- **`#pragma omp simd`** on the dot product, the gradient zero/scale/apply loops and
  every activation and loss. Inert unless the build enables OpenMP.
- **Derivative lookup.** Layers take only the forward functions and resolve both
  derivatives themselves, so a mismatched activation/derivative pair is unreachable.
- **`identity` activation**, and `identity` + `gelu` added to the derivative lookup.
- **A matplotlib plotting layer**: loss curves, architecture diagrams, weight
  distributions and heatmaps, decision boundaries, predicted-vs-actual and confusion
  matrices, 3-D loss landscapes, and a combined summary. Light and dark themes.
  `train(..., plot=True)` draws the loss curve on the way out.
- **`evaluate(X, Y)`** — dataset-wide mean loss, engine-side.
- **`set_weights` / `set_biases`** — there was previously no way to write parameters
  back at all.
- **`layers`** — machine-readable architecture, one dict per layer.

### Changed

- **`float` throughout the engine**, replacing `double`. Arrays cross the Python
  boundary as `float32`; `float64` input is accepted and cast.
- **Python `train()` delegates to `core::Network::train`.** The binding no longer
  runs a per-sample loop. A 4000-epoch XOR run through Python is bit-identical to the
  same run from `main.cpp`: `0.296616971 -> 0.000236376116`.
- `python/` is a package: the extension builds as `gradiex._core`, wrapped by
  `gradiex/__init__.py`.
- `computeLoss` calls the layer's configured loss instead of hardcoding squared error.
- Scratch pads are sized by `layerWidth` rather than `inputSize * width`.
- `pyproject.toml` and the extension report `0.0.3`, so `gradiex.__version__` finally
  matches the directory it ships in.
- `setup.py` lists `core/*.h` in `depends`; without it a header edit left a stale
  `.so` in place and the next run silently tested the previous engine.
- The C++ test suite was ported to the v0.0.3 API — 91 checks, passing under
  AddressSanitizer and UBSan.

### Fixed

- **The log-based losses clamped with `1e-12`, which rounds away in `float32`.**
  `1.0f - 1e-12f` *is* `1.0f`, so the upper clamp never bit; a saturated sigmoid gave
  `log(0)` and a division by zero, and `inf * 0` is `NaN`, which reaches every weight
  on the next update and never washes out. A two-moons run diverged at epoch 243 with
  the loss already down at 0.003. The guard is `1e-7f` now.
- **Both `fetchAssociatedDerivative` functions were non-inline in headers**, so a
  second translation unit failed to link. `core/loss.h` was also missing `#pragma once`.
- **`addHiddenLayer` / `addOutputLayer` did not bounds-check `layerIndex`** — an
  out-of-range index wrote past the end of the layer vector and leaked the layer.
  Both now throw `std::out_of_range`, and replacing a filled slot deletes the old layer.
- **`initializeAllWeightsAndBiases` dereferenced unfilled (null) slots.**
- **`Layer`'s five function-pointer members were indeterminate** after the default
  constructor. They now default to `nullptr`.
- **`Network()` set `numOutputLayers = 1` while resizing nothing**, so anything
  trusting the count read past the end of an empty vector. It reports `0` now.
- **A malformed OpenMP pragma.** `softmaxCrossEntropyDerivative` carried
  `reduction(+:vectorScratchPad[:size])` over a loop that overwrites the array rather
  than accumulating into it. Invisible while nothing enabled the pragmas; building the
  suite with `-fopenmp-simd` failed the fused-softmax test outright. The clause is
  gone, and `run_tests.sh --simd` now exercises that build.
- **`huber` did its arithmetic in `double`.** `0.5` rather than `0.5f` promoted the
  whole expression in an engine that is `float` everywhere else, which also stopped
  the loop vectorising — clang reports "loop not vectorized" for it under
  `-fopenmp-simd`. With the flag on, `core/` now compiles warning-free.

### Performance

Measured against v0.0.2 on an Apple M5, single-threaded, same harness for both
(`bench/run_bench.sh`), microseconds per sample, minimum of nine trials:

- **1.2–2.1×** from single precision and the structural changes alone.
- **1.2–3.2×** with `-fopenmp-simd`, which is what makes the SIMD directives apply.
- **1.7–5.5×** at batch 32, where the gradient-zeroing pass amortises.

The flag's contribution scales with layer width: nothing on `2→8×2→1`, 2.4× on
`784→128×2→10`. `-O3` made no measurable difference over `-O2` at any size.

### Known issues

- `std::expf` is not a standard name; `std::exp` is.
- Only `outputLayers[0]` back-propagates; extra output heads train silently wrong.
  The Python layer refuses to build them.
- `Network` has a destructor but no copy constructor, so a copy double-frees.
- The v0.0.2 zero-allocation assertion was not ported.

## [0.0.2] - 2026-09-12

Architecture rewrite. Same math, new memory model.

### Added

- **Python bindings** via pybind11. The package now exposes `gradiex.Network`,
  `gradiex.activations()`, `gradiex.losses()`, and `gradiex.initializers()`, and
  accepts anything NumPy can cast to `float64`.
- **New activations**: `gelu` (exact form) and `swish`.
- **New losses**: `mae` and `huber`.
- **Zero heap allocations on the hot path.** Forward, backward, and the SGD update
  touch only buffers sized once at `build()`. A test installs a global `operator new`
  counter and asserts zero allocations between `build()` and the end of a training run.
- **Flattened memory architecture.** All weights live in one contiguous
  `std::vector<double>`; all biases in a second. Layer caches, gradients, and
  pre-activations use the same flat layout, with layer offsets instead of pointers.
- **CPU loop unrolling.** The inner dot product uses four independent accumulators,
  summed as `(s0 + s1) + (s2 + s3)`, so the compiler can issue independent FMA chains
  without a serial dependency on one sum.
- **C++20** as the minimum standard.
- Bit-exactness test against a stored `0.0.1` run for supported regression and binary
  tasks on identical seeds and orderings.

### Changed

- Rewrote the engine from one `std::vector<double>` per neuron and one
  `std::vector<Neuron>` per layer to a flat, layer-contiguous layout.
- Layer connectivity is now index arithmetic over flat buffers; no pointers between
  `Neuron` objects.
- Hot loops read sequentially, improving cache locality. The speedup grows with layer
  width because the old engine paid for scattered weight vectors.
- The C++ API is lower-level than the Python API by design. The Python binding
  collects layer specs and constructs the engine only in `build()`, so invalid states
  raise Python exceptions instead of training silently wrong.
- Version intentionally rolled back to `0.0.2` to mark the architecture reset. The
  next feature-complete release will be `0.2.0`.

### Temporarily Removed / Not Yet Ported

These were present in `0.0.1` and are not yet in `0.0.2`. They are listed here so the
gap is explicit rather than discovered.

- Mini-batch training. `0.0.2` is per-sample SGD only.
- L2 weight decay.
- Inverted dropout.
- Train/validation split and `TrainHistory`.
- Early stopping.
- Model save/load. The flat layout makes this two `memcpy`s plus a header, planned.
- Fused softmax + categorical cross-entropy. Softmax is therefore unavailable as an
  output activation in this release.
- Multi-class output.
- Per-layer widths and per-layer dropout rates from the constructor.

### Fixed

- Documentation: the leaky ReLU formula in the file header read `max(0, ax)`, which
  has no leak for negative inputs. It is `x if x > 0 else a*x`.
- Documentation: Mean Absolute Error was listed under loss derivatives but was never
  implemented. It is now marked as unsupported in `0.0.1`; `mae` is implemented in
  `0.0.2`.

### Notes

- The five-argument `train(inputs, targets, epochs, learning_rate, log_interval)`
  from `0.0.1` is unchanged in the Python binding's `train()` wrapper, but the
  underlying engine is the new flat one.
- The C++ suite includes a zero-allocation assertion and a central-difference
  gradient check. The gradient check is what decides whether a change to the
  gradient path is correct.
- See [README.md](README.md) for the memory architecture, invariants, and roadmap.

## [0.0.1] - 2026-08-21

First public release.

### Added

- **Mini-batch training** with per-epoch shuffling. Gradients accumulate across a
  batch and apply once; the batch update equals the mean of the per-sample
  gradients exactly.
- **Seed control.** An optional seed argument fixes weight initialization,
  shuffling, and dropout masks. The seed in use is readable as `net.seed`.
- **Model save/load** in a versioned plain-text format. Values carry 17
  significant digits, so a reloaded network predicts bit-for-bit identically.
- **Multi-class classification** via softmax output fused with categorical
  cross-entropy, so the backward pass reduces to `p - t`.
- **Regularization**: L2 weight decay, inverted dropout on hidden layers, and a
  train/validation split with per-epoch loss history (`TrainHistory`).
- `TrainOptions` / `TrainHistory` structs and a `train(inputs, targets, options)`
  overload.
- `evaluate()` and `accuracy()` helpers.
- Test suite at `tests/test_gradiex.cpp`: 112 checks across 17 groups, including a
  central-difference gradient check over ten network configurations.
- MIT license, contribution guide, code of conduct, and CI.

### Changed

- **Activations and losses are now enums** rather than strings. Strings are parsed
  once at the API boundary instead of being compared per neuron per pass, which
  measured ~17% faster on a 64-128x3-10 network. The string-accepting
  constructors and `Neuron::activate` overloads are unchanged.
- `Neuron::forward` split into `compute_pre_activation` plus a layer-applied
  activation, so softmax can see every pre-activation before producing any output.

### Fixed

- Documentation: the leaky ReLU formula in the file header read `max(0, ax)`,
  which has no leak for negative inputs. It is `x if x > 0 else a*x`.
- Documentation: Mean Absolute Error was listed under loss derivatives but was
  never implemented. It is now marked as unsupported.

### Notes

- The original `train(inputs, targets, epochs, learning_rate, log_interval)`
  signature is unchanged and remains bit-identical to a manual fixed-order,
  one-sample-at-a-time loop. A test asserts this.
- No breaking API changes. Code written against the previous version compiles
  and behaves identically.