# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-21

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
