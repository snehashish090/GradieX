//
// Unit tests for core/layer.h : Layer, HiddenLayer, OutputLayer.
//
// Reference values in this file are hand-computed from the definitions in
// math.tex / the layer docs, never from the implementation's own output, so a
// test that fails means the implementation disagrees with the math.
//
// Each test runs in a forked child (see test_framework.h): an out-of-bounds
// write that kills the process shows up as [ CRASH ] on one line of the table
// instead of ending the run.
//
// The engine is float32. Reference values are computed here in double and
// compared with kTol (1e-6), which is the useful precision of a float32 result
// near 1.0; a handful of checks that are exact in binary floating point (zeros,
// halves, sums of small integers) use kExact.
//

#include "test_framework.h"
#include "../core/layer.h"

#include <numeric>
#include <vector>

namespace act = core::functions::activations;
namespace init = core::functions::initializers;
namespace loss = core::functions::loss;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// float32 carries ~7 decimal digits, so values near 1.0 resolve to ~1.2e-7.
static constexpr double kTol = 1e-6;
// For results that are exactly representable, so no slack is warranted.
static constexpr double kExact = 0.0;

// Constructor-shaped defaults so individual tests only name what they care
// about. The layer derives both derivatives itself, so they are not passed.
static core::Layer makeLayer(int width, int inputSize,
                             act::ActivationFunction activation = act::relu,
                             init::initFunction initializer = init::zeros,
                             loss::LossFunction lossFn = loss::mse)
{
    return core::Layer(width, inputSize, activation, initializer, lossFn);
}

static core::HiddenLayer makeHiddenLayer(int width, int inputSize,
                                         act::ActivationFunction activation)
{
    return core::HiddenLayer(width, inputSize, activation, init::zeros, loss::mse);
}

static core::OutputLayer makeOutputLayer(int width, int inputSize,
                                         act::ActivationFunction activation,
                                         loss::LossFunction lossFn)
{
    return core::OutputLayer(width, inputSize, activation, init::zeros, lossFn);
}

static void setRow(core::Layer& layer, int neuron, std::vector<float> row)
{
    for (int i = 0; i < static_cast<int>(row.size()); ++i)
        layer.neuronWeights[neuron * layer.inputSize + i] = row[i];
}

static double sampleStdDev(const std::vector<float>& v)
{
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    double acc = 0.0;
    for (float x : v) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

// ---------------------------------------------------------------------------
// A. construction and buffer allocation
// ---------------------------------------------------------------------------

TEST(Construction_StoresGeometryAndFunctionPointers)
{
    core::Layer layer = makeLayer(4, 3, act::sigmoid, init::he, loss::mae);
    ASSERT_EQ_INT(layer.layerWidth, 4);
    ASSERT_EQ_INT(layer.inputSize, 3);
    ASSERT_TRUE(layer.activationFunction == act::sigmoid);
    ASSERT_TRUE(layer.initialiser == init::he);
    ASSERT_TRUE(layer.lossFunction == loss::mae);
}

TEST(Construction_DerivesBothDerivativesFromTheForwardFunctions)
{
    // The constructor takes only the forward functions and resolves the
    // matching derivatives, so the two can never be paired up wrongly.
    core::Layer layer = makeLayer(2, 2, act::tanh, init::zeros, loss::huber);
    ASSERT_TRUE(layer.activationFunctionDerivative == act::tanhDerivative);
    ASSERT_TRUE(layer.lossFunctionDerivative == loss::huberDerivative);
}

TEST(Construction_ResizesForwardBuffers)
{
    core::Layer layer = makeLayer(4, 3);
    ASSERT_EQ_INT(layer.layerInputs.size(), 3);
    ASSERT_EQ_INT(layer.neuronOutputsUnactivated.size(), 4);
    ASSERT_EQ_INT(layer.neuronOutputsActivated.size(), 4);
    ASSERT_EQ_INT(layer.neuronDeltas.size(), 4);
    ASSERT_EQ_INT(layer.neuronWeights.size(), 12);
    ASSERT_EQ_INT(layer.neuronBiases.size(), 4);
}

TEST(Construction_ResizesGradientBuffers)
{
    // computeGradients() and updateWeightsAndBiases() index these directly, so
    // the constructor must size them alongside the forward buffers.
    core::Layer layer = makeLayer(4, 3);

    if (layer.neuronWeightGradients.size() != 12 || layer.neuronBiasGradients.size() != 4)
        FAIL_WITH_NOTE(
            "the constructor must resize the gradient buffers alongside the others",
            "neuronWeightGradients.size()=%zu (expected 12), neuronBiasGradients.size()=%zu (expected 4)",
            layer.neuronWeightGradients.size(), layer.neuronBiasGradients.size());
}

TEST(Construction_ScratchPadsHoldOneValuePerNeuron)
{
    // computeErrorSignal() writes layerWidth floats into each scratch pad, so
    // the pads are sized by layerWidth and not by inputSize. width 3 with
    // inputSize 0 is the case that catches an inputSize*width sizing.
    core::Layer layer = makeLayer(3, 0);
    if (layer.scratchPad1.size() < 3 || layer.scratchPad2.size() < 3)
        FAIL_WITH_NOTE(
            "scratch pads must be sized for layerWidth, independently of inputSize",
            "width=3 inputSize=0 -> scratchPad1.size()=%zu scratchPad2.size()=%zu (need >= 3)",
            layer.scratchPad1.size(), layer.scratchPad2.size());
}

TEST(DefaultConstructor_LeavesLayerEmptyAndFunctionsNull)
{
    // `Layer() noexcept = default` default-initialises the members, so the
    // function pointers need `= nullptr` initialisers or they hold garbage that
    // calculateOutputs() would jump to.
    core::Layer layer;
    ASSERT_EQ_INT(layer.layerWidth, 0);
    ASSERT_EQ_INT(layer.inputSize, 0);
    ASSERT_TRUE(layer.neuronWeights.empty());

    if (layer.activationFunction != nullptr
        || layer.activationFunctionDerivative != nullptr
        || layer.initialiser != nullptr
        || layer.lossFunction != nullptr
        || layer.lossFunctionDerivative != nullptr)
        FAIL_WITH_NOTE(
            "the function-pointer members need a default member initialiser, or they are "
            "indeterminate after Layer(); give each of them `= nullptr`",
            "activation=%p activationDerivative=%p initialiser=%p loss=%p lossDerivative=%p",
            (void*)layer.activationFunction, (void*)layer.activationFunctionDerivative,
            (void*)layer.initialiser, (void*)layer.lossFunction,
            (void*)layer.lossFunctionDerivative);
}

// ---------------------------------------------------------------------------
// B. derivative resolution
// ---------------------------------------------------------------------------

TEST(FetchAssociatedDerivative_ResolvesEveryTrainableActivation)
{
    // Every activation a layer can be built with must resolve, or the layer
    // stores a null derivative and segfaults on its first backward pass.
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::identity)   == act::identityDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::sigmoid)    == act::sigmoidDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::relu)       == act::reluDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::leakyRelu)  == act::leakyReluDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::tanh)       == act::tanhDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::swish)      == act::swishDerivative);
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::gelu)       == act::geluDerivative);
}

TEST(FetchAssociatedDerivative_SoftmaxIsDeliberatelyUnmapped)
{
    // softmax's derivative is a Jacobian; the usable form is fused with
    // categorical cross-entropy. Returning null is the documented contract, and
    // callers (the Python bindings) check for it and refuse to build.
    ASSERT_TRUE(act::fetchAssociatedDerivative(act::softmax) == nullptr);
}

TEST(FetchAssociatedDerivative_ResolvesEveryLoss)
{
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::mse)   == loss::mseDerivative);
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::mae)   == loss::maeDerivative);
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::huber) == loss::huberDerivative);
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::binaryCrossEntropy)
                == loss::binaryCrossEntropyDerivative);
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::crossEntropy)
                == loss::crossEntropyDerivative);
    ASSERT_TRUE(loss::fetchAssociatedDerivative(loss::softmaxCrossEntropy)
                == loss::softmaxCrossEntropyDerivative);
}

// ---------------------------------------------------------------------------
// C. initializeWeightsAndBiases
// ---------------------------------------------------------------------------

TEST(Initialize_ZerosInitializerClearsEveryWeight)
{
    core::Layer layer = makeLayer(4, 3, act::relu, init::zeros);
    std::fill(layer.neuronWeights.begin(), layer.neuronWeights.end(), 7.0f);

    layer.initializeWeightsAndBiases();

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeights[i], 0.0, kExact);
}

TEST(Initialize_ZeroesBiases)
{
    core::Layer layer = makeLayer(4, 3, act::relu, init::xavier);
    std::fill(layer.neuronBiases.begin(), layer.neuronBiases.end(), 3.5f);

    layer.initializeWeightsAndBiases();

    for (int i = 0; i < layer.layerWidth; ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiases[i], 0.0, kExact);
}

TEST(Initialize_XavierFillsEveryWeightWithFiniteValues)
{
    core::Layer layer = makeLayer(6, 5, act::relu, init::xavier);
    const float sentinel = -98765.0f;
    std::fill(layer.neuronWeights.begin(), layer.neuronWeights.end(), sentinel);

    layer.initializeWeightsAndBiases();

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
    {
        ASSERT_FINITE(layer.neuronWeights[i]);
        // sentinel still present => that slot was never written
        ASSERT_TRUE(layer.neuronWeights[i] != sentinel);
    }
}

TEST(Initialize_XavierStdDevMatchesFanInPlusFanOut)
{
    // W ~ N(0, sqrt(2 / (fanIn + fanOut)))
    const int width = 50, inputSize = 20;
    core::Layer layer = makeLayer(width, inputSize, act::relu, init::xavier);
    layer.initializeWeightsAndBiases();

    double expected = std::sqrt(2.0 / static_cast<double>(inputSize + width));
    double observed = sampleStdDev(layer.neuronWeights);
    // n = 1000 samples: sampling error on the std dev is ~2%, bounds are loose.
    ASSERT_TRUE(observed > expected * 0.75);
    ASSERT_TRUE(observed < expected * 1.25);
}

TEST(Initialize_HeStdDevMatchesFanIn)
{
    // W ~ N(0, sqrt(2 / fanIn)), fanIn = inputSize
    const int width = 50, inputSize = 20;
    core::Layer layer = makeLayer(width, inputSize, act::relu, init::he);
    layer.initializeWeightsAndBiases();

    double expected = std::sqrt(2.0 / static_cast<double>(inputSize));
    double observed = sampleStdDev(layer.neuronWeights);
    ASSERT_TRUE(observed > expected * 0.75);
    ASSERT_TRUE(observed < expected * 1.25);
}

TEST(Initialize_SeedMakesWeightsReproducible)
{
    // setSeed() is what makes a training run repeatable end to end; without it
    // the generator is seeded from std::random_device.
    core::Layer first = makeLayer(8, 4, act::relu, init::xavier);
    core::Layer second = makeLayer(8, 4, act::relu, init::xavier);

    init::setSeed(1234);
    first.initializeWeightsAndBiases();
    init::setSeed(1234);
    second.initializeWeightsAndBiases();

    for (size_t i = 0; i < first.neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, first.neuronWeights[i], second.neuronWeights[i], kExact);
}

// ---------------------------------------------------------------------------
// D. calculateOutputs
// ---------------------------------------------------------------------------

TEST(CalculateOutputs_ComputesWeightedSumPlusBias)
{
    // W = [[1,2,3],[4,5,6]], b = [0.5,-0.5], x = [2,1,0.5]
    // z0 = 2 + 2 + 1.5 + 0.5 = 6.0 ; z1 = 8 + 5 + 3 - 0.5 = 15.5
    core::Layer layer = makeLayer(2, 3, act::relu);
    setRow(layer, 0, {1.0f, 2.0f, 3.0f});
    setRow(layer, 1, {4.0f, 5.0f, 6.0f});
    layer.neuronBiases = {0.5f, -0.5f};

    const float input[3] = {2.0f, 1.0f, 0.5f};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], 6.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], 15.5, kExact);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 6.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 15.5, kExact);
}

TEST(CalculateOutputs_CachesItsOwnInput)
{
    // computeGradients() multiplies the delta by the vector that was fed to
    // calculateOutputs, so the layer must keep a copy of it.
    core::Layer layer = makeLayer(2, 3, act::relu);
    const float input[3] = {2.0f, 1.0f, 0.5f};
    layer.calculateOutputs(input);

    ASSERT_EQ_INT(layer.layerInputs.size(), 3);
    for (int i = 0; i < 3; ++i)
        ASSERT_NEAR_AT(i, layer.layerInputs[i], input[i], kExact);
}

TEST(CalculateOutputs_ReluZerosNegativePreActivationsButKeepsZ)
{
    // Same weights, x = [-1,-1,-1] -> z0 = -6 + 0.5 = -5.5 ; z1 = -15 - 0.5 = -15.5
    core::Layer layer = makeLayer(2, 3, act::relu);
    setRow(layer, 0, {1.0f, 2.0f, 3.0f});
    setRow(layer, 1, {4.0f, 5.0f, 6.0f});
    layer.neuronBiases = {0.5f, -0.5f};

    const float input[3] = {-1.0f, -1.0f, -1.0f};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], -5.5, kExact);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], -15.5, kExact);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.0, kExact);
}

TEST(CalculateOutputs_SigmoidMatchesReference)
{
    // W = [[1],[-1]], b = 0, x = [0.5] -> z = [0.5,-0.5]
    core::Layer layer = makeLayer(2, 1, act::sigmoid);
    setRow(layer, 0, {1.0f});
    setRow(layer, 1, {-1.0f});

    const float input[1] = {0.5f};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.6224593312018546, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.3775406687981454, kTol);
}

TEST(CalculateOutputs_SoftmaxNormalizesToOne)
{
    // weights are all zero, so z = b = [1,2,3]
    core::Layer layer = makeLayer(3, 2, act::softmax);
    layer.neuronBiases = {1.0f, 2.0f, 3.0f};

    const float input[2] = {0.3f, -0.7f};
    layer.calculateOutputs(input);

    double sum = static_cast<double>(layer.neuronOutputsActivated[0])
               + layer.neuronOutputsActivated[1] + layer.neuronOutputsActivated[2];
    ASSERT_NEAR(sum, 1.0, kTol);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.0900305731703805, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.2447284710547977, kTol);
    ASSERT_NEAR_AT(2, layer.neuronOutputsActivated[2], 0.6652409557748218, kTol);
}

TEST(CalculateOutputs_SingleNeuronWithWideInput)
{
    // Exercises row indexing when layerWidth < inputSize.
    // w = [0,0.5,1,...,3.5], x = 1 -> z = 0.5*(0+1+..+7) = 14, + 0.25 bias
    core::Layer layer = makeLayer(1, 8, act::relu);
    for (int i = 0; i < 8; ++i)
        layer.neuronWeights[i] = 0.5f * static_cast<float>(i);
    layer.neuronBiases[0] = 0.25f;

    std::vector<float> input(8, 1.0f);
    layer.calculateOutputs(input.data());

    ASSERT_NEAR(layer.neuronOutputsUnactivated[0], 14.25, kExact);
}

TEST(CalculateOutputs_IsIdempotentAcrossCalls)
{
    // A second call must overwrite, not accumulate into, the output buffers.
    core::Layer layer = makeLayer(2, 3, act::relu);
    setRow(layer, 0, {1.0f, 2.0f, 3.0f});
    setRow(layer, 1, {4.0f, 5.0f, 6.0f});
    layer.neuronBiases = {0.5f, -0.5f};

    const float first[3] = {2.0f, 1.0f, 0.5f};
    const float second[3] = {1.0f, 0.0f, 0.0f};
    layer.calculateOutputs(first);
    layer.calculateOutputs(second);

    // z0 = 1 + 0.5 = 1.5 ; z1 = 4 - 0.5 = 3.5
    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], 1.5, kExact);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], 3.5, kExact);
}

// ---------------------------------------------------------------------------
// E. computeGradients / zeroGradients / scaleGradients
// ---------------------------------------------------------------------------

TEST(ComputeGradients_BiasGradientEqualsDelta)
{
    core::Layer layer = makeLayer(3, 2);
    layer.neuronDeltas = {0.5f, -1.5f, 2.0f};

    layer.computeGradients();

    ASSERT_NEAR_AT(0, layer.neuronBiasGradients[0], 0.5, kExact);
    ASSERT_NEAR_AT(1, layer.neuronBiasGradients[1], -1.5, kExact);
    ASSERT_NEAR_AT(2, layer.neuronBiasGradients[2], 2.0, kExact);
}

TEST(ComputeGradients_WeightGradientEqualsDeltaTimesLayerInput)
{
    // dL/dW[n][i] = delta[n] * input[i], where input is the vector fed to
    // calculateOutputs (i.e. the previous layer's activations), not this
    // layer's own activations.
    core::Layer layer = makeLayer(2, 3, act::relu);
    setRow(layer, 0, {1.0f, 0.0f, 0.0f});
    setRow(layer, 1, {0.0f, 1.0f, 0.0f});

    const float input[3] = {1.0f, 2.0f, 3.0f};
    layer.calculateOutputs(input);
    layer.neuronDeltas = {0.5f, -1.5f};

    layer.computeGradients();

    for (int neuron = 0; neuron < layer.layerWidth; ++neuron)
        for (int i = 0; i < layer.inputSize; ++i)
        {
            double expected = layer.neuronDeltas[neuron] * input[i];
            double actual = layer.neuronWeightGradients[neuron * layer.inputSize + i];
            if (!testing::nearlyEqual(actual, expected, kTol))
                FAIL_WITH_NOTE(
                    "computeGradients() must multiply the delta by this layer's cached input "
                    "(layerInputs), not by its own activations",
                    "weightGradient[%d][%d]: actual %.12g, expected delta[%d]*input[%d] = %.12g",
                    neuron, i, actual, neuron, i, expected);
        }
}

TEST(ComputeGradients_WideInputStaysInsideItsBuffers)
{
    // layerWidth(1) < inputSize(6): indexing activations by input index would
    // read 6 values out of a 1-element vector. ASan reports that as a heap
    // overflow, so this test is the one to watch in the --asan build.
    core::Layer layer = makeLayer(1, 6, act::relu);

    std::vector<float> input(6);
    for (int i = 0; i < 6; ++i) input[i] = 1.0f + static_cast<float>(i);
    layer.calculateOutputs(input.data());
    layer.neuronDeltas[0] = 2.0f;

    layer.computeGradients();

    for (int i = 0; i < 6; ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeightGradients[i], 2.0 * input[i], kTol);
}

TEST(ComputeGradients_AccumulatesAcrossSamples)
{
    // 0.0.3 accumulates over a mini-batch rather than updating per sample, so
    // a second computeGradients() must add to the first, not replace it.
    core::Layer layer = makeLayer(1, 2, act::relu);

    const float firstInput[2] = {1.0f, 2.0f};
    layer.calculateOutputs(firstInput);
    layer.neuronDeltas[0] = 1.0f;
    layer.computeGradients();

    const float secondInput[2] = {3.0f, 4.0f};
    layer.calculateOutputs(secondInput);
    layer.neuronDeltas[0] = 2.0f;
    layer.computeGradients();

    // dW = 1*[1,2] + 2*[3,4] = [7, 10] ; db = 1 + 2 = 3
    ASSERT_NEAR_AT(0, layer.neuronWeightGradients[0], 7.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronWeightGradients[1], 10.0, kExact);
    ASSERT_NEAR(layer.neuronBiasGradients[0], 3.0, kExact);
}

TEST(ZeroGradients_ClearsBothBuffers)
{
    core::Layer layer = makeLayer(2, 3);
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 5.0f);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 5.0f);

    layer.zeroGradients();

    for (size_t i = 0; i < layer.neuronWeightGradients.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeightGradients[i], 0.0, kExact);
    for (size_t i = 0; i < layer.neuronBiasGradients.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiasGradients[i], 0.0, kExact);
}

TEST(ScaleGradients_ScalesBothBuffers)
{
    // The batch average is applied as scaleGradients(1/batchSize).
    core::Layer layer = makeLayer(2, 3);
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 8.0f);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 4.0f);

    layer.scaleGradients(0.25f);

    for (size_t i = 0; i < layer.neuronWeightGradients.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeightGradients[i], 2.0, kExact);
    for (size_t i = 0; i < layer.neuronBiasGradients.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiasGradients[i], 1.0, kExact);
}

// ---------------------------------------------------------------------------
// F. updateWeightsAndBiases
// ---------------------------------------------------------------------------

TEST(Update_AppliesGradientDescentStep)
{
    core::Layer layer = makeLayer(2, 3);
    std::fill(layer.neuronWeights.begin(), layer.neuronWeights.end(), 1.0f);
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 0.5f);
    std::fill(layer.neuronBiases.begin(), layer.neuronBiases.end(), 2.0f);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 1.0f);

    layer.updateWeightsAndBiases(0.1f);

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeights[i], 0.95, kTol);
    for (int i = 0; i < layer.layerWidth; ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiases[i], 1.9, kTol);
}

TEST(Update_ZeroLearningRateIsNoOp)
{
    core::Layer layer = makeLayer(2, 2);
    setRow(layer, 0, {0.25f, -0.5f});
    setRow(layer, 1, {1.5f, 2.0f});
    layer.neuronBiases = {0.75f, -1.25f};
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 99.0f);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 99.0f);

    layer.updateWeightsAndBiases(0.0f);

    ASSERT_NEAR_AT(0, layer.neuronWeights[0], 0.25, kExact);
    ASSERT_NEAR_AT(1, layer.neuronWeights[1], -0.5, kExact);
    ASSERT_NEAR_AT(2, layer.neuronWeights[2], 1.5, kExact);
    ASSERT_NEAR_AT(3, layer.neuronWeights[3], 2.0, kExact);
    ASSERT_NEAR_AT(0, layer.neuronBiases[0], 0.75, kExact);
    ASSERT_NEAR_AT(1, layer.neuronBiases[1], -1.25, kExact);
}

TEST(Update_OnFreshLayerIsMemorySafe)
{
    // The gradient buffers are sized by the constructor and zero-filled, so an
    // update before any backward pass is a no-op rather than a wild read.
    core::Layer layer = makeLayer(2, 2);
    layer.updateWeightsAndBiases(0.05f);

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_FINITE(layer.neuronWeights[i]);
}

// ---------------------------------------------------------------------------
// G. saturation in the log-based losses
//
// A sigmoid output saturates to exactly 1.0f (or 0.0f) in float32 long before
// it would in double. Everything here pins the losses staying finite at that
// point; the guard epsilon has to be representable either side of 1.0 in
// float32, which 1e-12 is not.
// ---------------------------------------------------------------------------

TEST(BinaryCrossEntropy_SaturatedPredictionStaysFinite)
{
    const float confidentlyWrong[1] = {1.0f};
    const float wantZero[1] = {0.0f};
    const float confidentlyWrongLow[1] = {0.0f};
    const float wantOne[1] = {1.0f};

    float high = loss::binaryCrossEntropy(wantZero, confidentlyWrong, 1);
    float low  = loss::binaryCrossEntropy(wantOne, confidentlyWrongLow, 1);

    if (!std::isfinite(high) || !std::isfinite(low))
        FAIL_WITH_NOTE(
            "the clamp epsilon must be representable either side of 1.0f in float32, or "
            "log(1 - p) is log(0) for a saturated prediction",
            "bce(y=0, p=1) = %.6g, bce(y=1, p=0) = %.6g", (double)high, (double)low);
    ASSERT_TRUE(high > 0.0f);
    ASSERT_TRUE(low > 0.0f);
}

TEST(BinaryCrossEntropyDerivative_SaturatedPredictionStaysFinite)
{
    const float saturated[1] = {1.0f};
    const float target[1] = {0.0f};
    float scratch[1] = {0.0f};

    loss::binaryCrossEntropyDerivative(target, saturated, scratch, 1);

    if (!std::isfinite(scratch[0]))
        FAIL_WITH_NOTE(
            "p*(1-p) is exactly 0 for an unclamped saturated prediction, so the derivative "
            "divides by zero",
            "bce'(y=0, p=1) = %.6g", (double)scratch[0]);
    ASSERT_TRUE(scratch[0] > 0.0f);   // push the prediction back down
}

TEST(OutputLayer_ErrorSignal_SaturatedSigmoidWithBceIsNotNaN)
{
    // The whole path, which is where this actually bit: a large pre-activation
    // makes the activation derivative 0 and the loss derivative infinite, and
    // inf * 0 is NaN -- which then reaches every weight on the next update and
    // never washes out.
    core::OutputLayer layer = makeOutputLayer(1, 2, act::sigmoid, loss::binaryCrossEntropy);
    layer.neuronOutputsUnactivated = {40.0f};   // sigmoid(40) rounds to exactly 1.0f
    layer.neuronOutputsActivated = {1.0f};

    const float target[1] = {0.0f};
    layer.computeErrorSignal(target);

    if (!std::isfinite(layer.neuronDeltas[0]))
        FAIL_WITH_NOTE(
            "a saturated sigmoid under binary cross-entropy produced a non-finite delta; "
            "one NaN here poisons every weight in the network",
            "delta = %.6g", (double)layer.neuronDeltas[0]);
}

TEST(CrossEntropy_ZeroProbabilityStaysFinite)
{
    const float target[2] = {1.0f, 0.0f};
    const float predicted[2] = {0.0f, 1.0f};
    float scratch[2] = {0.0f, 0.0f};

    float value = loss::crossEntropy(target, predicted, 2);
    loss::crossEntropyDerivative(target, predicted, scratch, 2);

    ASSERT_FINITE(value);
    ASSERT_FINITE(scratch[0]);
    ASSERT_FINITE(scratch[1]);
}

TEST(Training_SurvivesASaturatingSigmoidUnderBce)
{
    // End to end: drive a layer hard enough to saturate, then keep training. A
    // single non-finite delta would leave every weight NaN.
    core::HiddenLayer hidden = makeHiddenLayer(3, 2, act::tanh);
    core::OutputLayer output = makeOutputLayer(1, 3, act::sigmoid, loss::binaryCrossEntropy);
    for (auto& w : hidden.neuronWeights) w = 6.0f;      // saturate on purpose
    for (auto& w : output.neuronWeights) w = 12.0f;

    const float input[2] = {1.0f, 1.0f};
    const float target[1] = {0.0f};

    for (int step = 0; step < 50; ++step)
    {
        hidden.zeroGradients();
        output.zeroGradients();
        hidden.calculateOutputs(input);
        output.calculateOutputs(hidden.neuronOutputsActivated.data());
        output.computeErrorSignal(target);
        output.computeGradients();
        hidden.computeErrorSignal(output.layerWidth, output.neuronDeltas.data(),
                                  output.neuronWeights.data());
        hidden.computeGradients();
        hidden.updateWeightsAndBiases(0.5f);
        output.updateWeightsAndBiases(0.5f);
    }

    for (size_t i = 0; i < output.neuronWeights.size(); ++i)
        ASSERT_FINITE(output.neuronWeights[i]);
    for (size_t i = 0; i < hidden.neuronWeights.size(); ++i)
        ASSERT_FINITE(hidden.neuronWeights[i]);
}

// ---------------------------------------------------------------------------
// H. OutputLayer::computeErrorSignal
// ---------------------------------------------------------------------------

TEST(OutputLayer_ErrorSignal_MseWithSigmoid)
{
    // z = [0,0] -> a = [0.5,0.5], target = [1,0], N = 2
    // dL/da = 2(a-t)/N = [-0.5, 0.5] ; sigmoid'(0) = 0.25
    // delta = [-0.125, 0.125]
    core::OutputLayer layer = makeOutputLayer(2, 2, act::sigmoid, loss::mse);
    layer.neuronOutputsUnactivated = {0.0f, 0.0f};
    layer.neuronOutputsActivated = {0.5f, 0.5f};

    const float target[2] = {1.0f, 0.0f};
    layer.computeErrorSignal(target);

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], -0.125, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 0.125, kTol);
}

TEST(OutputLayer_ErrorSignal_ReluGateBlocksNegativePreActivation)
{
    // z = [-1,2] -> a = [0,2], target = [1,0], N = 2
    // dL/da = [-1, 2] ; relu'(z) = [0,1] -> delta = [0, 2]
    core::OutputLayer layer = makeOutputLayer(2, 2, act::relu, loss::mse);
    layer.neuronOutputsUnactivated = {-1.0f, 2.0f};
    layer.neuronOutputsActivated = {0.0f, 2.0f};

    const float target[2] = {1.0f, 0.0f};
    layer.computeErrorSignal(target);

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 2.0, kTol);
}

TEST(OutputLayer_ErrorSignal_PerfectPredictionGivesZeroDelta)
{
    core::OutputLayer layer = makeOutputLayer(3, 2, act::sigmoid, loss::mse);
    layer.neuronOutputsUnactivated = {0.4f, -0.2f, 1.1f};
    layer.neuronOutputsActivated = {0.25f, 0.5f, 0.75f};

    const float target[3] = {0.25f, 0.5f, 0.75f};
    layer.computeErrorSignal(target);

    for (int i = 0; i < 3; ++i)
        ASSERT_NEAR_AT(i, layer.neuronDeltas[i], 0.0, kExact);
}

TEST(OutputLayer_ErrorSignal_BinaryCrossEntropyWithSigmoid)
{
    // z = [0] -> a = [0.5], target = [1], N = 1
    // dL/da = (a-t)/(a(1-a))/N = -0.5/0.25 = -2 ; sigmoid'(0) = 0.25
    // delta = -0.5   (== a - t, the familiar BCE+sigmoid simplification)
    core::OutputLayer layer = makeOutputLayer(1, 2, act::sigmoid, loss::binaryCrossEntropy);
    layer.neuronOutputsUnactivated = {0.0f};
    layer.neuronOutputsActivated = {0.5f};

    const float target[1] = {1.0f};
    layer.computeErrorSignal(target);

    ASSERT_NEAR(layer.neuronDeltas[0], -0.5, kTol);
}

TEST(OutputLayer_ErrorSignal_UsesDistinctScratchPads)
{
    // Loss derivative and activation derivative must not share a buffer, or the
    // second call overwrites the first before they are multiplied together.
    core::OutputLayer layer = makeOutputLayer(3, 1, act::sigmoid, loss::mse);
    layer.neuronOutputsUnactivated = {0.0f, 0.0f, 0.0f};
    layer.neuronOutputsActivated = {0.5f, 0.5f, 0.5f};

    const float target[3] = {1.0f, 0.0f, 1.0f};
    layer.computeErrorSignal(target);

    // dL/da = 2(a-t)/3 = [-1/3, 1/3, -1/3] ; sigmoid'(0) = 0.25
    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], -1.0 / 12.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.0 / 12.0, kTol);
    ASSERT_NEAR_AT(2, layer.neuronDeltas[2], -1.0 / 12.0, kTol);
}

TEST(OutputLayer_ErrorSignal_FusedSoftmaxCrossEntropyIsProbabilityMinusTarget)
{
    // identity activation + softmaxCrossEntropy is the only trainable route to
    // a softmax classifier: the loss derivative consumes the logits directly
    // and returns softmax(z) - y, and identity'(z) = 1 leaves it untouched.
    // This is the pairing the Python bindings steer callers to.
    core::OutputLayer layer = makeOutputLayer(3, 1, act::identity, loss::softmaxCrossEntropy);
    layer.neuronOutputsUnactivated = {1.0f, 2.0f, 3.0f};
    layer.neuronOutputsActivated = {1.0f, 2.0f, 3.0f};   // identity

    const float target[3] = {0.0f, 1.0f, 0.0f};
    layer.computeErrorSignal(target);

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.0900305731703805, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 0.2447284710547977 - 1.0, kTol);
    ASSERT_NEAR_AT(2, layer.neuronDeltas[2], 0.6652409557748218, kTol);
}

// ---------------------------------------------------------------------------
// I. HiddenLayer::computeErrorSignal
// ---------------------------------------------------------------------------

TEST(HiddenLayer_ErrorSignal_SingleNextNeuron)
{
    // this layer: width 2, inputSize 3, relu, z = [1,-1] -> relu' = [1,0]
    // next layer: width 1, inputSize 2, W = [0.5,-0.5], delta = [1]
    // expected: [1*0.5*1, 1*(-0.5)*0] = [0.5, 0]
    core::HiddenLayer layer = makeHiddenLayer(2, 3, act::relu);
    layer.neuronOutputsUnactivated = {1.0f, -1.0f};

    std::vector<float> nextWeights = {0.5f, -0.5f};
    std::vector<float> nextDeltas = {1.0f};

    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.5, kExact);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 0.0, kExact);
}

TEST(HiddenLayer_ErrorSignal_ReluGateZerosDeadNeuron)
{
    core::HiddenLayer layer = makeHiddenLayer(2, 2, act::relu);
    layer.neuronOutputsUnactivated = {-1.0f, 1.0f};

    std::vector<float> nextWeights = {0.5f, 0.5f};
    std::vector<float> nextDeltas = {2.0f};

    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.0, kExact);
}

TEST(HiddenLayer_ErrorSignal_OverwritesPreviousDeltas)
{
    // The delta buffer is zeroed before the propagation sum, so a second call
    // must not add to the first.
    core::HiddenLayer layer = makeHiddenLayer(2, 2, act::relu);
    layer.neuronOutputsUnactivated = {1.0f, 1.0f};
    layer.neuronDeltas = {99.0f, 99.0f};

    std::vector<float> nextWeights = {0.5f, 0.5f};
    std::vector<float> nextDeltas = {2.0f};

    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());
    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 1.0, kExact);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.0, kExact);
}

TEST(HiddenLayer_ErrorSignal_AppliesDerivativeExactlyOnce)
{
    // width == inputSize == 2 so the weight stride is right either way; this
    // isolates the derivative multiply from the indexing.
    // next layer: width 2, inputSize 2, W = [[1,2],[3,4]], delta = [1,1]
    // sum0 = 1*1 + 1*3 = 4 ; sum1 = 1*2 + 1*4 = 6 ; sigmoid'(0) = 0.25
    // expected: [1.0, 1.5]
    core::HiddenLayer layer = makeHiddenLayer(2, 2, act::sigmoid);
    layer.neuronOutputsUnactivated = {0.0f, 0.0f};

    std::vector<float> nextWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> nextDeltas = {1.0f, 1.0f};

    layer.computeErrorSignal(2, nextDeltas.data(), nextWeights.data());

    if (!testing::nearlyEqual(layer.neuronDeltas[0], 1.0, kTol))
        FAIL_WITH_NOTE(
            "the activation-derivative multiply must sit outside the next-neuron loop, or it "
            "is applied once per next-layer neuron instead of once per neuron",
            "delta[0]: actual %.12g, expected 1.0", (double)layer.neuronDeltas[0]);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.5, kTol);
}

TEST(HiddenLayer_ErrorSignal_StridesNextWeightsByOwnWidth)
{
    // width 2, inputSize 3: the next layer's row stride is ITS inputSize, which
    // equals THIS layer's layerWidth (2) -- not this layer's inputSize (3).
    // relu with z > 0 makes the derivative 1, isolating the indexing.
    // next layer: width 2, inputSize 2, W = [[1,2],[3,4]], delta = [1,1]
    // expected: [1+3, 2+4] = [4, 6]
    core::HiddenLayer layer = makeHiddenLayer(2, 3, act::relu);
    layer.neuronOutputsUnactivated = {1.0f, 1.0f};

    std::vector<float> nextWeights = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> nextDeltas = {1.0f, 1.0f};

    layer.computeErrorSignal(2, nextDeltas.data(), nextWeights.data());

    if (!testing::nearlyEqual(layer.neuronDeltas[0], 4.0, kTol))
        FAIL_WITH_NOTE(
            "nextLayerWeights must be indexed with stride this->layerWidth; using "
            "this->inputSize also reads past the end of the next layer's weights",
            "delta[0]: actual %.12g, expected 4.0", (double)layer.neuronDeltas[0]);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 6.0, kTol);
}

// ---------------------------------------------------------------------------
// J. multi-layer integration
// ---------------------------------------------------------------------------

TEST(Integration_TwoLayerForwardChain)
{
    // hidden: width 3, inputSize 2, relu, W = [[1,0],[0,1],[1,1]], b = 0
    //   x = [0.5,0.25] -> z = a = [0.5, 0.25, 0.75]
    // output: width 1, inputSize 3, sigmoid, W = [1,1,1], b = 0
    //   z = 1.5 -> a = sigmoid(1.5)
    core::HiddenLayer hidden = makeHiddenLayer(3, 2, act::relu);
    setRow(hidden, 0, {1.0f, 0.0f});
    setRow(hidden, 1, {0.0f, 1.0f});
    setRow(hidden, 2, {1.0f, 1.0f});

    core::OutputLayer output = makeOutputLayer(1, 3, act::sigmoid, loss::mse);
    setRow(output, 0, {1.0f, 1.0f, 1.0f});

    const float input[2] = {0.5f, 0.25f};
    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());

    ASSERT_NEAR_AT(0, hidden.neuronOutputsActivated[0], 0.5, kExact);
    ASSERT_NEAR_AT(1, hidden.neuronOutputsActivated[1], 0.25, kExact);
    ASSERT_NEAR_AT(2, hidden.neuronOutputsActivated[2], 0.75, kExact);
    ASSERT_NEAR(output.neuronOutputsUnactivated[0], 1.5, kExact);
    ASSERT_NEAR(output.neuronOutputsActivated[0], 0.8175744761936437, kTol);
}

TEST(Integration_ForwardBackwardUpdateLowersLoss)
{
    // One gradient step on a fixed (input, target) pair must not increase the
    // loss. Mirrors main.cpp's geometry: 2 -> hidden 2 -> output 1.
    core::HiddenLayer hidden = makeHiddenLayer(2, 2, act::sigmoid);
    core::OutputLayer output = makeOutputLayer(1, 2, act::sigmoid, loss::mse);
    setRow(hidden, 0, {0.5f, -0.3f});
    setRow(hidden, 1, {0.2f, 0.8f});
    setRow(output, 0, {0.7f, -0.4f});

    const float input[2] = {0.1f, 0.2f};
    const float target[1] = {1.0f};

    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());
    double lossBefore = loss::mse(target, output.neuronOutputsActivated.data(), 1);

    output.computeErrorSignal(target);
    output.computeGradients();
    hidden.computeErrorSignal(output.layerWidth,
                              output.neuronDeltas.data(),
                              output.neuronWeights.data());
    hidden.computeGradients();
    hidden.updateWeightsAndBiases(0.5f);
    output.updateWeightsAndBiases(0.5f);

    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());
    double lossAfter = loss::mse(target, output.neuronOutputsActivated.data(), 1);

    ASSERT_FINITE(lossAfter);
    if (lossAfter > lossBefore + 1e-9)
        FAIL_WITH_NOTE(
            "a single descent step moved uphill, so at least one gradient has the wrong "
            "value or sign",
            "loss before %.12g, loss after %.12g", lossBefore, lossAfter);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    return testing::runAll("core/layer.h", argc > 1 ? argv[1] : nullptr);
}
