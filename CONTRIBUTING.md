# Contributing to GradieX

Thanks for your interest. GradieX is a small, dependency-free C++ neural network
library, and the goal is to keep it readable enough that someone can learn
backpropagation by reading the source.

## Getting started

```bash
git clone https://github.com/<owner>/GradieX.git
cd GradieX

# build and run the examples
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o gradiex && ./gradiex

# build and run the test suite
g++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/test_gradiex.cpp -o test_gradiex
./test_gradiex
```

The suite must print `112 passed, 0 failed` and exit `0`. It exits non-zero on any
failure, so it works directly in CI or a pre-commit hook.

There is no build system on purpose. `main.cpp` is one translation unit with no
dependencies beyond the C++17 standard library, and `tests/test_gradiex.cpp`
`#include`s it directly (renaming its `main`), so there is no duplicated source to
keep in sync.

## Before you open a pull request

1. **The test suite passes.** All 112 checks, not just the ones near your change.
2. **Zero compiler warnings** under `-Wall -Wextra -pedantic`.
3. **Clean under sanitizers:**
   ```bash
   g++ -std=c++17 -O1 -g -fsanitize=address,undefined tests/test_gradiex.cpp -o san_test
   ./san_test
   ```
4. **New behaviour comes with a test.** If it can be wrong, it needs a check.
5. **Update the docs.** The file header in `main.cpp` and `README.md` both document
   the API; if you change it, change them. Add a `CHANGELOG.md` entry.

## Touching the backward pass

If you modify anything in the gradient path, the gradient check in Test 1 is the
thing that decides whether you got it right. It compares analytic gradients against
central differences across ten network configurations, to roughly $10^{-11}$ absolute.
It is not a smoke test — it is the reason this library can be trusted.

Add a configuration to the `cases` list in Test 1 if you add an activation, a loss,
or a new path through `accumulate_gradients`.

## Invariants

These are not style preferences. Breaking one produces a library that trains
plausibly and is silently wrong.

1. **Gradients accumulate; no weight moves until `apply_gradients()`.** This is what
   makes mini-batching exact, and it is what guarantees the backward pass reads
   pre-update weights when it propagates
   $\partial L / \partial a_j^{(l-1)} = \sum_i \delta_i^{(l)} w_{ij}^{(l)}$. Updating
   a layer's weights before propagating through it corrupts every layer behind it.

2. **`input_cache` must hold the input from *this* forward pass.** Weight gradients
   are meaningless against a stale cache.

3. **`Neuron::output` is the activation *before* the dropout mask;
   `Layer::output_vector` is the value *after* it.** The sigmoid and tanh derivatives
   are written in terms of the unmasked output, so masking `Neuron::output` silently
   corrupts them. With dropout off the two are equal, which is exactly why this is
   easy to break without noticing.

4. **Softmax is layer-wide and fused with categorical cross-entropy.**
   `Neuron::activate` and `Neuron::activation_derivative` throw on softmax rather
   than return a wrong answer. Softmax's Jacobian is a dense $n \times n$ matrix and
   the scalar-in / scalar-out derivative signature cannot express it. The supported
   path uses $\partial L / \partial \mathbf{z} = \mathbf{p} - \mathbf{y}$.

5. **`Activation` and `Loss` are enums on the hot path.** Strings are parsed once at
   the API boundary. Do not push a string comparison back into `Layer::forward` or
   the per-neuron loops.

6. **The five-argument `train()` stays bit-identical** to a manual fixed-order,
   one-sample-at-a-time loop. A test asserts this. It is the backward-compatible
   entry point.

7. **Bump the model format version** in `save()`/`load()` if the file layout changes.

## Code style

Match the surrounding code rather than any external guide:

- Four-space indent, no tabs. Opening brace on the same line.
- `PascalCase` for types, `snake_case` for functions, variables, and members.
- **Spell things out.** This codebase uses `gradient_wrt_output`, `neuron_index`,
  `pre_activation` — not `g`, `i`, `z`. Loop counters included. The verbosity is
  deliberate; it is what makes the math readable next to the code.
- Comments explain *why*, not *what*. The math is in the file header; inline
  comments are for decisions a reader would otherwise question.
- No new dependencies. Standard library only.

## Good first contributions

The roadmap in `README.md` lists the open work. In rough order of value:

- **Momentum / Adam optimizers.** Plain SGD is the current ceiling on hard problems.
- **Per-layer widths from the constructor.** `load()` already reads per-layer shapes,
  so the engine handles non-uniform networks; only the constructor is restrictive.
- **Early stopping** driven by `TrainHistory`, which already tracks the best epoch.
- **Per-layer dropout rates.**

Smaller but genuinely useful: more worked examples, a CMake option for people who
want one, or feature normalization helpers (the engine assumes roughly unit-scale
inputs and does not scale for you).

## Reporting bugs

Open an issue with the compiler and version, the OS, a minimal `main()` that
reproduces it, and what you expected instead. If it is a numerical problem, include
the seed — every run is reproducible from one.

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
