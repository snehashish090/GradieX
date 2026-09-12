# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on version ordering.** `0.0.2` is an architecture rewrite that temporarily
> drops features present in `0.0.1`. It is listed first because it is the newest
> release, despite the lower SemVer number. `0.0.1` remains the last feature-complete
> release.

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