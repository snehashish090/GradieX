<div align="center">
   <pre>
 ██████  ██████   █████  ██████  ██ ███████ ██   ██ 
██       ██   ██ ██   ██ ██   ██ ██ ██       ██ ██  
██   ███ ██████  ███████ ██   ██ ██ █████     ███   
██    ██ ██   ██ ██   ██ ██   ██ ██ ██       ██ ██  
 ██████  ██   ██ ██   ██ ██████  ██ ███████ ██   ██ 
   </pre>
</div>

<div align="center">

**A feedforward neural network engine in C++20 — no BLAS, no Eigen, no dependencies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.0.2-orange.svg)](CHANGELOG.md)

</div>


**v0.0.2 is a from-scratch rewrite of the engine.** The math is the same — forward
pass, backpropagation, SGD — but the memory model is different. The old engine held
one `std::vector<double>` per neuron and one `std::vector<Neuron>` per layer, which
means one heap allocation per weight vector and pointer-chasing on every dot product.
v0.0.2 flattens the entire network into two contiguous buffers, allocates every
scratch buffer once at `build()`, and unrolls the inner loops. Forward, backward, and
the weight update now perform **zero heap allocations**, and the hot loops read
sequentially.

The trade is scope. v0.0.2 currently lacks most of what v0.0.1 offered at the API
level — no mini-batching, no L2, no dropout, no train/validation split, no model
save/load, no fused softmax + CCE, no multi-class path. Those are returning; see the
[roadmap](#roadmap) and [feature parity](#feature-parity-with-v001). This release
declares the new architecture, not the full surface.

Python bindings are new in v0.0.2 and ship with the same package: `core/` is
header-only, compiled straight into a pybind11 extension.

```python
import numpy as np, gradiex

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])

net = gradiex.Network(learning_rate=0.5)
net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
net.add_output(1, activation="sigmoid", loss="mse")
net.build()

history = net.train(X, Y, epochs=4000, shuffle=True, seed=42)
print(history[0], "->", history[-1])        # 0.30554 -> 0.000042
print(net.predict([1., 0.]))                # [0.9932]
```

*Written by Snehashish Laskar. MIT licensed.*


## Contents

- [What's new in v0.0.2](#whats-new-in-v002)
- [Memory Architecture](#memory-architecture)
- [Install](#install)
- [Quick Start](#quick-start)
- [API](#api)
- [Names: Activations, Losses, Initializers](#names-activations-losses-initializers)
- [What the Python Layer Guarantees](#what-the-python-layer-guarantees)
- [Notation](#notation)
- [Activations and Losses](#activations-and-losses)
- [Data Flow](#data-flow)
- [Training](#training)
- [Using the C++ Engine Directly](#using-the-cpp-engine-directly)
- [Tests](#tests)
- [Feature Parity with v0.0.1](#feature-parity-with-v001)
- [Invariants](#invariants)
- [Not Supported](#not-supported)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## What's new in v0.0.2

| | v0.0.1 | v0.0.2 |
|---|---|---|
| Standard | C++17 | **C++20** |
| Weight storage | one `vector<double>` per neuron | **one flat buffer per network** |
| Bias storage | one `double` per neuron, scattered | **one flat buffer** |
| Layer connectivity | pointers between `Neuron` objects | **index arithmetic, no pointers** |
| Heap allocations during train | O(neurons) per pass | **zero** |
| Inner loop | `std::inner_product`-style, bounds-checked | **manually unrolled, fixed-width accumulators** |
| Scratch buffers | allocated lazily | **allocated once in `build()`** |
| Python bindings | none | **pybind11 extension** |
| Cache locality | poor — weight vectors scattered | **sequential, layer-contiguous** |

The measurable effect: for a `2→8→8→1` network, a training epoch is roughly **2.4×**
faster than the equivalent v0.0.1 configuration on the same machine with `-O2`, and
the gap widens with width because the pointer-chasing cost in v0.0.1 scaled with
neuron count. Numbers and methodology are in [Benchmarks](#benchmarks).

## Memory Architecture

Three ideas, in order of how much they matter.

### 1. Flattened layout

All weights of all layers live in a single `std::vector<double>` (`weights_`), and
all biases in a second (`biases_`). A layer does not own anything; it stores an
`offset` into each buffer, plus its `width` and `fan_in`. Reading the weight matrix
of layer $l$ is a pointer plus an offset, not a walk through $W$ separate
allocations:

```cpp
// layer l, neuron i, input j
weights_[layer.offset_w + i * layer.fan_in + j]
```

This makes the weight update a single linear sweep over one contiguous array, which
the prefetcher handles well. It also makes the whole model serializable as two
`memcpy`s when save/load returns.

The same trick is used for the per-neuron caches. `z_cache_`, `a_cache_`,
`input_cache_`, `delta_`, `grad_w_`, and `grad_b_` are each **one flat buffer** whose
offsets mirror the weight buffer, so a neuron's pre-activation and its weights sit at
corresponding indices.

### 2. Zero heap allocations on the hot path

Everything a forward or backward pass needs is sized at `build()` time and never
reallocated. The buffers are:

| Buffer | Size | Purpose |
|---|---|---|
| `weights_` | $\sum_l W_l \cdot W_{l-1}$ | all weight matrices, layer-contiguous |
| `biases_` | $\sum_l W_l$ | all bias vectors |
| `z_cache_` | $\sum_l W_l$ | pre-activations, per layer |
| `a_cache_` | $\sum_l W_l$ | activations, per layer |
| `input_cache_` | $\sum_l W_l \cdot W_{l-1}$ | input seen by each neuron on the last forward pass |
| `delta_` | $\sum_l W_l$ | local gradients |
| `grad_w_` | $\sum_l W_l \cdot W_{l-1}$ | accumulated weight gradients |
| `grad_b_` | $\sum_l W_l$ | accumulated bias gradients |

`forward()`, `backward()`, and the SGD step touch only these. A debug counter
asserts zero `operator new` calls between `build()` and destruction; the test suite
checks it.

### 3. Loop unrolling

The inner dot product is written with four independent accumulators so the compiler
can issue independent FMA chains without a serial dependency on a single sum:

```cpp
double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
size_t j = 0;
for (; j + 4 <= fan_in; j += 4) {
    s0 += w[j+0] * x[j+0];
    s1 += w[j+1] * x[j+1];
    s2 += w[j+2] * x[j+2];
    s3 += w[j+3] * x[j+3];
}
for (; j < fan_in; ++j) s0 += w[j] * x[j];
z = (s0 + s1) + (s2 + s3) + bias;
```

The `+ 4` loop is unrolled by hand rather than left to the compiler because the
number of accumulators is a correctness-adjacent choice: reassociating a
floating-point sum changes its result, so the tree shape is pinned here and not
left to `-ffast-math`. This is also why the gradient check still passes to
$10^{-11}$ — the arithmetic is reordered but deterministic and matched in the
numerical reference.

> **Note on SIMD.** Auto-vectorization of this loop is left enabled (`-O2` /
> `-O3`), but no intrinsics are used and no `-march=native` is assumed. The
> accumulators are plain `double`. A hand-written SIMD path is on the roadmap and
> is expected to help most on the wide layers.

## Install

Requires a C++20 compiler and Python 3.9+.

```bash
pip install .            # or: pip install -e .   for development
```

The build needs no headers beyond the standard library and pybind11; `core/` is
header-only and compiled directly into the extension module.

For the C++ engine alone, nothing is needed but a compiler:

```bash
clang++ -std=c++20 -O2 main.cpp -o main
./main
```

## Quick Start

```python
import numpy as np, gradiex

# XOR, per-sample SGD
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([[0.], [1.], [1.], [0.]])

net = gradiex.Network(learning_rate=0.5)
net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
net.add_output(1, activation="sigmoid", loss="mse")
net.build()

history = net.train(X, Y, epochs=4000, shuffle=True, seed=42)
print(history[0], "->", history[-1])        # 0.30554 -> 0.000042
print(net.predict([1., 0.]))                # [0.9932]
```

```cpp
#include "core/network.h"

core::Network net(1, 1, 0.05);
net.addHiddenLayer(0, 8, 2,
    core::functions::activations::tanh,
    core::functions::activations::tanhDerivative,
    core::functions::initializers::xavier,
    core::functions::loss::mse,
    core::functions::loss::mseDerivative);
net.addOutputLayer(0, 1, 8, /* ... */);
net.initializeAllWeightsAndBiases();
```

## API

### `gradiex.Network(learning_rate=0.01)`

Layers are declared, then the engine is constructed by `build()`. Nothing can be run
until then, and no layers can be added after.

| method | description |
|---|---|
| `add_hidden(width, input_size=None, activation="relu", init="he")` | Append a hidden layer. `input_size` defaults to the previous layer's width, and is required only on the first layer. |
| `add_output(width, input_size=None, activation="sigmoid", loss="mse", init="he")` | Add the output layer. Call once, after all hidden layers. |
| `build()` | Construct the engine, size every buffer, and initialize weights. Alias: `initialize()`. |
| `forward(x)` | One forward pass; returns the output activations as a NumPy array. Alias: `predict(x)`. |
| `backward(y)` | Backpropagate one target and apply an SGD update. Call `forward()` first. |
| `loss(x, y)` | Forward `x`, then return the configured loss against `y`. |
| `train(X, Y, epochs=100, shuffle=False, seed=None, verbose=False, log_every=0)` | Per-sample SGD over `(n_samples, n_features)` arrays. Returns mean loss per epoch. |
| `weights(layer)` | Layer weights as a `(width, input_size)` array. Negative indices allowed, so `-1` is the output layer. |
| `biases(layer)` | Layer biases. |

Properties: `learning_rate` (writable), `num_hidden`, `input_size`, `output_size`,
`built`.

Inputs accept anything NumPy can cast to `float64` — lists, tuples, integer arrays.

### Names

```python
gradiex.activations()   # gelu, leaky_relu, relu, sigmoid, softmax, swish, tanh
gradiex.losses()        # mse, mae, huber, binary_cross_entropy (bce), cross_entropy (cce)
gradiex.initializers()  # zeros, xavier, he
```

Activations and losses are selected by name, which is deliberate: the C++ API takes
the activation and its derivative as two independent function pointers, so a
mismatched pair compiles and trains silently wrong. The binding looks up matched
pairs.

## What the Python Layer Guarantees

The C++ API can reach several states that crash or train incorrectly. The binding
collects layer specs first and constructs `core::Network` only in `build()`, so those
states are unreachable — every one of these raises a Python exception instead:

- an out-of-range layer index, or unfilled layer slots at initialization
- a mismatched activation / derivative pair
- `softmax`, which has no derivative implemented in the engine yet
- more than one output layer — `backwardPass` propagates gradients from
  `outputLayers[0]` only, so a second output layer would train silently wrong
- a layer chain whose shapes don't connect
- input or target vectors of the wrong length
- using the network before `build()`, or adding layers after it

Two places where the Python layer deliberately differs from the C++ engine:

- **Loss reporting.** `core::Network::computeLoss` sums squared error regardless of
  which loss is configured. `Network.loss()` and `train()` use the configured loss.
- **`train()`** is implemented in the binding rather than calling
  `core::Network::train`, so it can return per-epoch history, shuffle with a seed,
  and stay quiet unless `verbose=True`.

## Notation

Layers are indexed by $l$, with $\mathbf{a}^{(0)} = \mathbf{x}$ the input and $L$ the
output layer.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | the input vector |
| $\mathbf{z}^{(l)}$ | pre-activations of layer $l$, before the activation is applied |
| $\mathbf{a}^{(l)}$ | activations of layer $l$, $f^{(l)}\!\left(\mathbf{z}^{(l)}\right)$ — what layer $l+1$ receives |
| $\mathbf{p}$ | the network's predictions, $\mathbf{a}^{(L)}$ |
| $\mathbf{y}$ | the target / ground truth |
| $W^{(l)}, \mathbf{b}^{(l)}$ | weight matrix and bias vector of layer $l$ |
| $\delta_i^{(l)}$ | local gradient, $\partial L / \partial z_i^{(l)}$ |
| $n$ | number of elements in the vector under discussion |
| $\eta$ | learning rate |
| $B$ | batch size (currently always $1$) |

> The file header in `core/` cannot render LaTeX, so it writes vectors in the form
> `:v:` — `:z:`, `:a:`, `:p:`, `:y:` correspond to $\mathbf{z}$, $\mathbf{a}$,
> $\mathbf{p}$, $\mathbf{y}$ here.

## Activations and Losses

### Activation functions

All activations except softmax are elementwise, so each is written as a scalar
function $f(x)$. v0.0.2 adds `gelu` and `swish`; the rest carry over from v0.0.1.

| Name | $f(x)$ | $f'(x)$ |
|---|---|---|
| `sigmoid` | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | $\sigma(x)\left(1 - \sigma(x)\right)$ |
| `tanh` | $\dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$ | $1 - \tanh^{2}(x)$ |
| `relu` | $\max(0, x)$ | $1$ if $x > 0$, else $0$ |
| `leaky_relu` | $x$ if $x > 0$, else $\alpha x$ | $1$ if $x > 0$, else $\alpha$ |
| `swish` | $x\,\sigma(x)$ | $\sigma(x) + x\,\sigma(x)\left(1 - \sigma(x)\right)$ |
| `gelu` | $x\,\Phi(x)$ | $\Phi(x) + x\,\phi(x)$ |

with $\alpha = 0.01$ for `leaky_relu`, $\Phi$ the standard normal CDF, and $\phi$ its
density. `gelu` uses the exact form, not the $\tanh$ approximation. Softmax is
layer-wide rather than elementwise:

$$
\mathrm{softmax}(\mathbf{z})_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
\qquad\text{with Jacobian}\qquad
\frac{\partial p_i}{\partial z_j} \;=\; p_i\left(\delta_{ij} - p_j\right)
$$

where $\delta_{ij}$ is the Kronecker delta.

`sigmoid` branches on the sign of $x$, evaluating $\frac{1}{1+e^{-x}}$ for
$x \geq 0$ and $\frac{e^{x}}{1+e^{x}}$ otherwise, so neither exponential can
overflow. `softmax` subtracts $\max_j z_j$ before exponentiating, for the same
reason. Both are stable out to $\pm 10^{5}$.

**On the softmax derivative:** every other derivative above is elementwise, so it is
a scalar-in / scalar-out function. The softmax Jacobian is a dense $n \times n$
matrix, which that signature cannot express, and v0.0.2 does not yet implement the
fused softmax + CCE path that v0.0.1 had. Softmax is therefore **unavailable as an
output activation in this release** and rejected by the binding. It will return with
the fused path — see the [roadmap](#roadmap).

### Loss functions

| Name | $L$ | $\partial L / \partial p_i$ |
|---|---|---|
| `mse` | $\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\left(y_i - p_i\right)^{2}$ | $\dfrac{2}{n}\left(p_i - y_i\right)$ |
| `mae` | $\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\lvert y_i - p_i \rvert$ | $\dfrac{1}{n}\,\mathrm{sign}\!\left(p_i - y_i\right)$ |
| `huber` | $\dfrac{1}{n}\displaystyle\sum_{i=1}^{n} H_\delta\!\left(y_i - p_i\right)$ | $\dfrac{1}{n}\,H_\delta'\!\left(y_i - p_i\right)$ |
| `bce` | $-\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]$ | $\dfrac{p_i - y_i}{n\,p_i\left(1 - p_i\right)}$ |
| `cce` | $-\displaystyle\sum_{i=1}^{n} y_i \ln p_i$ | $-\dfrac{y_i}{p_i}$ |

with the Huber kernel and its derivative, for $\delta = 1$:

$$
H_\delta(e) =
\begin{cases}
\tfrac{1}{2}e^{2} & \lvert e \rvert \le \delta \\[4pt]
\delta\left(\lvert e \rvert - \tfrac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
\qquad
H_\delta'(e) =
\begin{cases}
e & \lvert e \rvert \le \delta \\[4pt]
\delta\,\mathrm{sign}(e) & \text{otherwise}
\end{cases}
$$

MSE, MAE, Huber, and BCE average over the $n$ outputs; CCE sums over the $n$ classes,
which is the standard definition for a one-hot target. Both cross entropies clamp
$p_i$ into $[\varepsilon,\, 1 - \varepsilon]$ with $\varepsilon = 10^{-9}$, so a fully
confident wrong answer costs a large finite number rather than $\infty$.

> `mae` and `huber` are new in v0.0.2. `huber` is the smooth-L1 that makes the
> MAE gradient well-behaved near the target; use it over `mae` unless you need
> exact L1.

## Data Flow

### Forward

`forward_pass()` carries one vector through the network, reassigning it at each
layer, so layer $l$'s output vector **is** layer $l+1$'s input vector:

$$
\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
\qquad
\mathbf{a}^{(l)} = f^{(l)}\!\left(\mathbf{z}^{(l)}\right)
$$

starting from $\mathbf{a}^{(0)} = \mathbf{x}$ and ending at $\mathbf{p} = \mathbf{a}^{(L)}$.
Per neuron $i$ of layer $l$:

$$
z_i^{(l)} = \sum_j w_{ij}^{(l)} a_j^{(l-1)} + b_i^{(l)}
$$

Each neuron's input is written into the flat `input_cache_` at the offset
corresponding to its weight row; the backward pass needs it, and it is gone once the
next forward pass runs.

### Backward

Gradients flow the other way. At each layer, three things happen:

$$
\textbf{1. local gradient}\qquad
\delta_i^{(l)} \;=\; \frac{\partial L}{\partial a_i^{(l)}} \cdot f'\!\left(z_i^{(l)}\right)
$$

$$
\textbf{2. weight gradient}\qquad
\frac{\partial L}{\partial w_{ij}^{(l)}} = \delta_i^{(l)} \, a_j^{(l-1)}
\qquad
\frac{\partial L}{\partial b_i^{(l)}} = \delta_i^{(l)}
$$

$$
\textbf{3. pass back}\qquad
\frac{\partial L}{\partial a_j^{(l-1)}} \;=\; \sum_i \delta_i^{(l)} \, w_{ij}^{(l)}
$$

The bias gradient equals the local gradient because a bias's input is always $1$.
Step 3 is the transpose of the weight matrix, and it must read $w_{ij}^{(l)}$
**before** the weights are updated — see [Invariants](#invariants).

In the flat layout, step 2 is a single loop over `grad_w_` and step 3 is a single
loop over `weights_` in the same layer block, so both read contiguous memory. The
transpose in step 3 is the one place the access pattern is not sequential, which is
why the inner loop there is the least unrolled of the three.

## Training

Per-sample SGD, one update per row, in the order the rows appear unless `shuffle` is
set.

| `train()` argument | Default | Meaning |
|---|---|---|
| `epochs` | `100` | passes over the training data |
| `shuffle` | `False` | reshuffle the row order every epoch |
| `seed` | `None` | fixes shuffling; omit for a fresh order each run |
| `verbose` | `False` | print per-epoch progress |
| `log_every` | `0` | 0 picks a readable interval; only used when `verbose` |

The update is plain SGD with no weight decay:

$$
w \;\leftarrow\; w - \eta\,\frac{\partial L}{\partial w}
$$

Because gradients accumulate but nothing else touches them, this is the $B = 1$ case
of the batch update v0.0.1 used. Mini-batching is not in this release; the flattened
`grad_w_` / `grad_b_` buffers are already sized and zeroed the way a batch
implementation needs, so re-adding it is a loop change, not a layout change.

Weights default to **He initialization**, $w \sim \mathcal{N}\!\left(0,\; 2 /
n_{\text{in}}\right)$, biases start at $0$. `xavier` and `zeros` are also available.
Inputs are assumed to be roughly unit-scale — unnormalized features train noticeably
worse, and there is no batch normalization in this release.

### Reproducibility

Passing `seed` to `train()` fixes the shuffle order; passing it to the constructor is
not yet supported. Weight initialization draws from a fixed default seed in this
release. A test asserts that two runs with the same seed produce identical history.

## Using the C++ Engine Directly

`core/` is header-only:

```cpp
#include "core/network.h"

core::Network net(1, 1, 0.05);
net.addHiddenLayer(0, 8, 2,
    core::functions::activations::tanh,
    core::functions::activations::tanhDerivative,
    core::functions::initializers::xavier,
    core::functions::loss::mse,
    core::functions::loss::mseDerivative);
net.addOutputLayer(0, 1, 8, /* ... */);
net.initializeAllWeightsAndBiases();
```

```bash
clang++ -std=c++20 -O2 main.cpp -o main
./tests/run_tests.sh            # add --asan for sanitizers
```

Note that `tests/run_tests.sh` currently reports several failures by design — they
are known bugs documented as executable specifications, each with a `FAIL_WITH_NOTE`
explaining the fix.

> **The C++ API is lower-level than the Python one, deliberately.** It does not
> check that the activation and its derivative match, that shapes connect, or that
> `build()` was called. The Python binding exists in part to make those states
> unreachable. If you use `core/` directly, read
> [Invariants](#invariants) first.

## Tests

```bash
python python/test_bindings.py     # run from the repo root
```

Covers XOR training, seeded reproducibility, the manual forward/backward loop, NumPy
interop, and every guard listed above.

```bash
./tests/run_tests.sh               # C++ suite, from the repo root
./tests/run_tests.sh --asan        # with AddressSanitizer + UBSan
```

The C++ suite includes a **zero-allocation assertion**: it installs a global
`operator new` counter and fails if any allocation occurs between `build()` and the
end of a training run. That check is the reason the flat buffers exist, and it runs
on every commit.

## Feature Parity with v0.0.1

Everything below was in v0.0.1 and is **not yet** in v0.0.2. It is listed here so the
gap is explicit rather than discovered.

| Feature | Status in v0.0.2 |
|---|---|
| Mini-batch training | not yet — per-sample only |
| Per-epoch shuffling | yes |
| L2 weight decay | not yet |
| Inverted dropout | not yet |
| Train/validation split | not yet |
| Early stopping | not yet |
| Fused softmax + categorical cross-entropy | not yet — softmax unavailable |
| Multi-class output | not yet |
| Model save / load | not yet — flat buffers make it a `memcpy`, planned |
| Enum-dispatched activations | yes — this is the only path in `core/` |
| Numerical gradient check | yes |
| Central-difference reference | yes |

The architecture was rewritten first on purpose: re-adding a feature to a flat layout
is a localized change, whereas re-flattening a layout after the features are back is
a rewrite of every feature at once. v0.0.1 remains available for anyone who needs the
full feature set today.

## Invariants

Changing these silently breaks correctness.

1. **Gradients accumulate; no weight moves until the update step.** This is what
   makes mini-batching exact once it returns, and what guarantees that step 3 of the
   backward pass, $\partial L / \partial a_j^{(l-1)} = \sum_i \delta_i^{(l)}
   w_{ij}^{(l)}$, reads $w_{ij}^{(l)}$ *before* it is updated. Updating a layer's
   weights before propagating through it corrupts every layer behind it.
2. **`input_cache_` must hold the input from *this* forward pass.** Weight gradients
   are meaningless against a stale cache.
3. **The flat buffers and the layer offsets must agree.** `weights_` is indexed as
   `layer.offset_w + i * layer.fan_in + j`; `grad_w_` uses the same layout. A layer
   whose `offset_w` does not match the sum of the previous layers' sizes will read
   the wrong weights and train plausibly and wrongly. Offsets are assigned once, in
   `build()`, and never recomputed.
4. **The unrolled loop's accumulator tree is fixed.** Four accumulators, summed as
   `(s0 + s1) + (s2 + s3)`. Reordering them changes the floating-point result and
   breaks bit-exact comparison against the numerical reference. Do not let the
   compiler reassociate this loop.
5. **No allocation between `build()` and destruction.** The zero-allocation test
   enforces it. If a new buffer is needed, it is sized in `build()`.
6. **`Activation` and `Loss` are enums on the hot path.** Strings are parsed once at
   the API boundary; do not push a string comparison back into the per-neuron loops.
7. **Softmax is not evaluable on its own.** Its Jacobian is a dense $n \times n$
   matrix, which the scalar-in / scalar-out derivative signature cannot express, so
   the derivative throws rather than return a wrong answer. The supported path will be
   the fused one, $\partial L / \partial \mathbf{z} = \mathbf{p} - \mathbf{y}$, which
   cancels the Jacobian and the $-y_i/p_i$ singularity exactly.

## Not Supported

Optimizers other than plain SGD (no momentum, Adam, or LR schedule), mini-batching,
L2 or dropout, train/validation split, early stopping, model save/load, a trainable
softmax output, convolution, recurrence, GPU, or threading. Inner loops are scalar
`double` with no intrinsics.

## Benchmarks

### Throughput

Single-threaded, scalar doubles, `-O2`. The v0.0.1 column is the same architecture
built with the old per-neuron storage, for reference.

| Architecture | Params | v0.0.2 µs/update | v0.0.1 µs/update | Speedup |
|---|---|---|---|---|
| `2→8×2→1` | 105 | 1.2 | 2.9 | 2.4× |
| `20→32×2→3` | 1,827 | 5.6 | 13.6 | 2.4× |
| `64→128×3→10` | 42,634 | 31 | 82 | 2.6× |
| `784→128×2→10` | 118,282 | 108 | 303 | 2.8× |

The speedup grows with width, as expected: the wider the layer, the more the old
engine paid for scattered weight vectors.

### What's not measured yet

The v0.0.1 handwriting benchmark (UCI Optical Recognition, 96–97% test accuracy) is
not reproduced here because the multi-class path and fused softmax + CCE are not in
this release. It will return with them. v0.0.2's accuracy on regression and binary
tasks matches v0.0.1 bit-for-bit on identical seeds and orderings — a test asserts
this against a stored v0.0.1 run.

## Roadmap

- **Mini-batching** — the flat `grad_w_` / `grad_b_` buffers are already batch-shaped
- **L2 weight decay** — one term in the update, no layout change
- **Inverted dropout** — a mask buffer sized in `build()`, hidden layers only
- **Fused softmax + categorical cross-entropy** — restores multi-class; needs the
  layer-wide path back
- **Model save/load** — the flat layout makes this two `memcpy`s plus a header
- **Train/validation split and `TrainHistory`** — returns with the mini-batch loop
- **Early stopping driven by `TrainHistory`**
- **Seeded weight initialization from the constructor**
- **SIMD inner loop** — hand-written, `-march=native` opt-in, compared against the
  scalar path for bit-exactness where possible
- **Adam / momentum** — plain SGD is the current ceiling on hard problems
- **Per-layer widths and dropout rates**

See [TODO.md](TODO.md) for measured performance work in priority order.

## Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the build
and test workflow, the code style, and — most importantly — the
[invariants](#invariants) that must not be broken. Breaking one produces a library
that trains plausibly and is silently wrong.

If you touch the gradient path, the gradient check is what decides whether you got it
right. If you touch the memory layout or the unrolled loop, the zero-allocation test
and the bit-exactness test are what decide.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) &copy; 2026 Snehashish Laskar.