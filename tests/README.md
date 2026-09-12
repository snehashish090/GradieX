# Test suite

Unit tests for `core/layer.h` (`Layer`, `HiddenLayer`, `OutputLayer`) and
`core/network.h` (`Network`).

```sh
./tests/run_tests.sh            # both suites, plain build
./tests/run_tests.sh --asan     # + AddressSanitizer/UBSan (finds the out-of-bounds
                                #   accesses that the plain build silently survives)
./tests/run_tests.sh Gradients  # only tests whose name contains "Gradients"
```

| file | what it covers |
| --- | --- |
| `test_framework.h` | the harness (~200 lines, no dependencies) |
| `test_layer.cpp` | 35 tests: construction, initializers, `calculateOutputs`, `computeGradients`, `updateWeightsAndBiases`, both `computeErrorSignal` overloads |
| `test_network.cpp` | 24 tests: layer registration, `forwardPass`, `backwardPass`, `computeLoss`, `train`, teardown |

## How it works

**Every test runs in its own forked process.** A segfault therefore shows as
`[ CRASH ]` on one row of the table instead of ending the run — which is what
you want when the code under test is crashing. Each child also gets a
10-second `alarm()`, so a hang counts as a failure too.

Expected values are computed in the test from the definitions — the comment
above each test shows the arithmetic — never captured from the
implementation's output. A failing test therefore means the code disagrees with
the math, not with itself.

Tests that aren't about buffer allocation call `ensureGradientBuffers()` /
build a fully-populated network first, so one defect doesn't cascade into
every other result.

## Current results

`59 tests: 53 pass, 5 fail, 1 crash`. Everything numerical passes — forward
chaining, output and hidden deltas, the gradient step, and convergence over
epochs. The six remaining failures are all missing guards rather than wrong
math:

| Test | Gap |
| --- | --- |
| `Construction_ScratchPadsHoldAtLeastOneValuePerNeuron` | `layer.h:65-66` — pads are sized `inputSize*width`, but `computeErrorSignal()` writes `layerWidth` doubles. The size shouldn't involve `inputSize`; `resize(width)` is the right amount. |
| `DefaultConstructor_LeavesLayerEmptyAndFunctionsNull` | `layer.h:37` — `Layer() noexcept = default` leaves the five function pointers indeterminate. They need `= nullptr` initialisers. |
| `DefaultConstructor_LayerCountsMatchAllocation` | `network.h:16` — `Network()` sets `numOutputLayers = 1` but resizes nothing, so `backwardPass()`'s unconditional `outputLayers[0]` reads an empty vector. |
| `AddHiddenLayer_OutOfRangeIndexIsRejected` | `network.h:40,58` — `layerVector[layerIndex] = layer` is unchecked; a bad index writes past the end (ASan confirms) and leaks the layer. Use `.at()` or throw. |
| `InitializeAllWeightsAndBiases_SkipsUnfilledSlots` | `network.h:63-66` — dereferences null slots when a network was sized for more layers than were added. The destructor already tolerates this (`delete nullptr`); this loop should too. |
| `ComputeLoss_AgreesWithTheConfiguredLossFunction` | `network.h` — `computeLoss()` hardcodes a squared-error sum while `backwardPass()` differentiates the layer's `lossFunction`, so the number printed during training isn't the quantity being minimised (0.875 vs 0.292 on a 3-wide output). |

## Regression tests worth knowing about

- `Train_MoreEpochsThanSamplesStaysInBounds` — pins the crash where `train()`
  indexed the dataset by the epoch counter and ran off the end on
  `epoch == numSamples`.
- `ForwardPass_RecordsEachLayersOwnInput` — pins `layerInputs` actually being
  written, without which every weight gradient is zero and only the biases learn.
- `HiddenLayer_ErrorSignal_AppliesDerivativeExactlyOnce` — the derivative
  multiply in `HiddenLayer::computeErrorSignal` used to be indented as though it
  were inside the inner loop. It wasn't, and this test keeps a future re-indent
  from changing that.

## Adding a test

```cpp
TEST(Thing_DoesWhatTheMathSays)
{
    // W = [[1,2]], b = [0.5], x = [1,1] -> z = 3.5
    core::Layer layer = makeLayer(1, 2, act::relu, act::reluDerivative);
    ...
    ASSERT_NEAR(layer.neuronOutputsUnactivated[0], 3.5, kTol);
}
```

Assertions: `ASSERT_TRUE`, `ASSERT_FALSE`, `ASSERT_EQ_INT`, `ASSERT_NEAR`,
`ASSERT_NEAR_AT(index, ...)`, `ASSERT_FINITE`, and `FAIL_WITH_NOTE(note, fmt, ...)`
for a failure that should explain itself to whoever reads the output.
