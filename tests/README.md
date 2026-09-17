# Test suite

Unit tests for `core/layer.h` (`Layer`, `HiddenLayer`, `OutputLayer`) and
`core/network.h` (`Network`).

```sh
./tests/run_tests.sh            # both suites, plain build
./tests/run_tests.sh --asan     # + AddressSanitizer/UBSan (finds the out-of-bounds
                                #   accesses that the plain build silently survives)
./tests/run_tests.sh --simd     # + -fopenmp-simd, so the `#pragma omp simd`
                                #   directives in core/ actually apply
./tests/run_tests.sh Gradients  # only tests whose name contains "Gradients"
```

| file | what it covers |
| --- | --- |
| `test_framework.h` | the harness (~200 lines, no dependencies) |
| `test_layer.cpp` | 49 tests: construction, derivative resolution, initializers, `calculateOutputs`, gradient accumulate/zero/scale, `updateWeightsAndBiases`, both `computeErrorSignal` overloads, saturation in the log-based losses |
| `test_network.cpp` | 42 tests: layer registration, `forwardPass`, `accumulateGradients`/`applyGradients`, `computeLoss`, `trainBatch`, `train`, teardown |

## How it works

**Every test runs in its own forked process.** A segfault therefore shows as
`[ CRASH ]` on one row of the table instead of ending the run — which is what
you want when the code under test is crashing. Each child also gets a
10-second `alarm()`, so a hang counts as a failure too.

Expected values are computed in the test from the definitions — the comment
above each test shows the arithmetic — never captured from the
implementation's output. A failing test therefore means the code disagrees with
the math, not with itself.

### float32 tolerances

The engine is `float` end to end, so reference values computed here in `double`
agree only to about seven digits. Two constants express that:

- `kTol` (`1e-6`) — the useful precision of a float32 result near 1.0. Anything
  compared against a hand-computed transcendental uses this.
- `kExact` (`0.0`) — for results that are exactly representable in binary
  floating point (zeros, halves, sums of small integers) and for "this value did
  not change", which is checked by snapshotting the buffer rather than comparing
  against a decimal literal. `0.7f != 0.7`, so a literal would fail for the
  wrong reason.

## Current results

`91 tests: 91 pass`, in every mode: plain, `--asan`, `--simd`, and both together.

**`--simd` is not optional before shipping.** Unrecognised `#pragma omp` directives
are discarded silently — `-Wall -Wextra` says nothing — so a malformed one is
invisible until a build enables them. `softmaxCrossEntropyDerivative` carried an
array-section `reduction` clause over a loop that overwrites rather than accumulates;
the suite passed without the flag and failed the fused-softmax test with it.

These suites were written against the 0.0.2 engine and ported to 0.0.3. The port
was mostly mechanical — `double` → `float`, the layer constructors dropping their
derivative arguments now that the engine resolves those itself, and
`backwardPass()` splitting into `zeroAllGradients()` / `accumulateGradients()` /
`applyGradients()`. Most of the old suite's failures were fixed by the 0.0.3
rewrite itself (scratch-pad sizing, gradient-buffer allocation, the cached
`layerInputs`, the next-layer weight stride, `computeLoss` using the configured
loss function). Four were still open and were fixed alongside the port:

| Test | Fix |
| --- | --- |
| `DefaultConstructor_LeavesLayerEmptyAndFunctionsNull` | `layer.h` — the five function-pointer members got `= nullptr` initialisers; `Layer() noexcept = default` left them indeterminate. |
| `DefaultConstructor_LayerCountsMatchAllocation` | `network.h` — `Network()` set `numOutputLayers = 1` but resized nothing, so `accumulateGradients()`'s unconditional `outputLayers[0]` read an empty vector. It now reports 0. |
| `AddHiddenLayer_OutOfRangeIndexIsRejected` | `network.h` — `layerVector[layerIndex] = layer` was unchecked; a bad index wrote past the end (ASan confirmed) and leaked the layer. Both adders now bounds-check and throw `std::out_of_range`. |
| `InitializeAllWeightsAndBiases_SkipsUnfilledSlots` | `network.h` — dereferenced null slots when a network was sized for more layers than were added. It now skips them, as the destructor already did. |

## Regression tests worth knowing about

- `Train_MoreEpochsThanSamplesStaysInBounds` — pins the crash where `train()`
  indexed the dataset by the epoch counter and ran off the end on
  `epoch == numSamples`.
- `ForwardPass_RecordsEachLayersOwnInput` — pins `layerInputs` actually being
  written, without which every weight gradient is zero and only the biases learn.
- `HiddenLayer_ErrorSignal_AppliesDerivativeExactlyOnce` — the derivative
  multiply in `HiddenLayer::computeErrorSignal` must stay outside the
  next-neuron loop, or it is applied once per next-layer neuron.
- `HiddenLayer_ErrorSignal_StridesNextWeightsByOwnWidth` — the next layer's row
  stride is *this* layer's `layerWidth`, not its `inputSize`.
- `TrainBatch_AveragesGradientsOverTheBatch` — a batch applies the mean of the
  per-sample gradients, so the same sample twice in one batch equals one
  single-sample step.
- `Train_ShuffleVisitsEverySampleOncePerEpoch` — the shuffle must permute, not
  resample: one full-batch epoch is the same update either way.
- `OutputLayer_ErrorSignal_FusedSoftmaxCrossEntropyIsProbabilityMinusTarget` —
  pins the only trainable route to a softmax classifier (`identity` activation
  with `softmax_cross_entropy`), which is what the Python bindings steer callers
  to.

## Adding a test

```cpp
TEST(Thing_DoesWhatTheMathSays)
{
    // W = [[1,2]], b = [0.5], x = [1,1] -> z = 3.5
    core::Layer layer = makeLayer(1, 2, act::relu);
    ...
    ASSERT_NEAR(layer.neuronOutputsUnactivated[0], 3.5, kExact);
}
```

Assertions: `ASSERT_TRUE`, `ASSERT_FALSE`, `ASSERT_EQ_INT`, `ASSERT_NEAR`,
`ASSERT_NEAR_AT(index, ...)`, `ASSERT_FINITE`, and `FAIL_WITH_NOTE(note, fmt, ...)`
for a failure that should explain itself to whoever reads the output.

## Python

The bindings and the plotting layer have their own checks, run from the release
root after `python3 setup.py build_ext --inplace`:

```sh
python3 python/test_bindings.py    # engine bindings, guards, C++/Python parity
python3 python/test_plotting.py    # every diagram, headless (Agg)
```
