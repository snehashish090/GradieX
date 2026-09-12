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

// Constructor-shaped defaults so individual tests only name what they care about.
static core::Layer makeLayer(int width, int inputSize,
                             act::ActivationFunction activation = act::relu,
                             act::ActivationFunctionDerivative activationDerivative = act::reluDerivative,
                             init::initFunction initializer = init::zeros,
                             loss::LossFunction lossFn = loss::mse,
                             loss::LossFunctionDerivative lossDerivative = loss::mseDerivative)
{
    return core::Layer(width, inputSize, activation, activationDerivative,
                       initializer, lossFn, lossDerivative);
}

static core::HiddenLayer makeHiddenLayer(int width, int inputSize,
                                         act::ActivationFunction activation,
                                         act::ActivationFunctionDerivative activationDerivative)
{
    return core::HiddenLayer(width, inputSize, activation, activationDerivative,
                             init::zeros, loss::mse, loss::mseDerivative);
}

static core::OutputLayer makeOutputLayer(int width, int inputSize,
                                         act::ActivationFunction activation,
                                         act::ActivationFunctionDerivative activationDerivative,
                                         loss::LossFunction lossFn,
                                         loss::LossFunctionDerivative lossDerivative)
{
    return core::OutputLayer(width, inputSize, activation, activationDerivative,
                             init::zeros, lossFn, lossDerivative);
}

// The Layer constructor does NOT size the gradient buffers (see
// Construction_ResizesGradientBuffers). Tests that are about something else
// call this first so one allocation bug doesn't mask every other result.
static void ensureGradientBuffers(core::Layer& layer)
{
    layer.neuronWeightGradients.assign(
        static_cast<size_t>(layer.layerWidth) * static_cast<size_t>(layer.inputSize), 0.0);
    layer.neuronBiasGradients.assign(static_cast<size_t>(layer.layerWidth), 0.0);
}

static void setRow(core::Layer& layer, int neuron, std::vector<double> row)
{
    for (int i = 0; i < static_cast<int>(row.size()); ++i)
        layer.neuronWeights[neuron * layer.inputSize + i] = row[i];
}

static double sampleStdDev(const std::vector<double>& v)
{
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    double acc = 0.0;
    for (double x : v) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

static constexpr double kTol = 1e-9;

// ---------------------------------------------------------------------------
// A. construction and buffer allocation
// ---------------------------------------------------------------------------

TEST(Construction_StoresGeometryAndFunctionPointers)
{
    core::Layer layer = makeLayer(4, 3, act::sigmoid, act::sigmoidDerivative,
                                  init::he, loss::mae, loss::maeDerivative);
    ASSERT_EQ_INT(layer.layerWidth, 4);
    ASSERT_EQ_INT(layer.inputSize, 3);
    ASSERT_TRUE(layer.activationFunction == act::sigmoid);
    ASSERT_TRUE(layer.activationFunctionDerivative == act::sigmoidDerivative);
    ASSERT_TRUE(layer.initialiser == init::he);
    ASSERT_TRUE(layer.lossFunction == loss::mae);
    ASSERT_TRUE(layer.lossFunctionDerivative == loss::maeDerivative);
}

TEST(Construction_ResizesForwardBuffers)
{
    core::Layer layer = makeLayer(4, 3);
    ASSERT_EQ_INT(layer.neuronOutputsUnactivated.size(), 4);
    ASSERT_EQ_INT(layer.neuronOutputsActivated.size(), 4);
    ASSERT_EQ_INT(layer.neuronDeltas.size(), 4);
    ASSERT_EQ_INT(layer.neuronWeights.size(), 12);
    ASSERT_EQ_INT(layer.neuronBiases.size(), 4);
}

TEST(Construction_ResizesGradientBuffers)
{
    core::Layer layer = makeLayer(4, 3);

    if (layer.neuronWeightGradients.size() != 12 || layer.neuronBiasGradients.size() != 4)
        FAIL_WITH_NOTE(
            "the constructor must resize the gradient buffers alongside the others",
            "neuronWeightGradients.size()=%zu (expected 12), neuronBiasGradients.size()=%zu (expected 4)",
            layer.neuronWeightGradients.size(), layer.neuronBiasGradients.size());
}

TEST(Construction_ScratchPadsHoldAtLeastOneValuePerNeuron)
{
    // computeErrorSignal() writes layerWidth doubles into each scratch pad, so
    // the pads must be at least layerWidth long. They are sized
    // inputSize*width, which is short whenever inputSize == 0.
    core::Layer layer = makeLayer(3, 0);
    if (layer.scratchPad1.size() < 3 || layer.scratchPad2.size() < 3)
        FAIL_WITH_NOTE(
            "scratch pads are sized inputSize*width; they must be sized for layerWidth",
            "width=3 inputSize=0 -> scratchPad1.size()=%zu scratchPad2.size()=%zu (need >= 3)",
            layer.scratchPad1.size(), layer.scratchPad2.size());
}

TEST(DefaultConstructor_LeavesLayerEmptyAndFunctionsNull)
{
    // `Layer() noexcept = default` leaves the function-pointer members
    // indeterminate, so a default-constructed layer holds garbage that
    // calculateOutputs() would jump to. They need `= nullptr` initialisers.
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
            "the function-pointer members have no default member initialiser, so they are "
            "indeterminate after Layer(); give each of them `= nullptr`",
            "activation=%p activationDerivative=%p initialiser=%p loss=%p lossDerivative=%p",
            (void*)layer.activationFunction, (void*)layer.activationFunctionDerivative,
            (void*)layer.initialiser, (void*)layer.lossFunction,
            (void*)layer.lossFunctionDerivative);
}

// ---------------------------------------------------------------------------
// B. initializeWeightsAndBiases
// ---------------------------------------------------------------------------

TEST(Initialize_ZerosInitializerClearsEveryWeight)
{
    core::Layer layer = makeLayer(4, 3, act::relu, act::reluDerivative, init::zeros);
    std::fill(layer.neuronWeights.begin(), layer.neuronWeights.end(), 7.0);

    layer.initializeWeightsAndBiases();

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeights[i], 0.0, kTol);
}

TEST(Initialize_ZeroesBiases)
{
    core::Layer layer = makeLayer(4, 3, act::relu, act::reluDerivative, init::xavier);
    std::fill(layer.neuronBiases.begin(), layer.neuronBiases.end(), 3.5);

    layer.initializeWeightsAndBiases();

    for (int i = 0; i < layer.layerWidth; ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiases[i], 0.0, kTol);
}

TEST(Initialize_XavierFillsEveryWeightWithFiniteValues)
{
    core::Layer layer = makeLayer(6, 5, act::relu, act::reluDerivative, init::xavier);
    const double sentinel = 1e300;
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
    core::Layer layer = makeLayer(width, inputSize, act::relu, act::reluDerivative, init::xavier);
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
    core::Layer layer = makeLayer(width, inputSize, act::relu, act::reluDerivative, init::he);
    layer.initializeWeightsAndBiases();

    double expected = std::sqrt(2.0 / static_cast<double>(inputSize));
    double observed = sampleStdDev(layer.neuronWeights);
    ASSERT_TRUE(observed > expected * 0.75);
    ASSERT_TRUE(observed < expected * 1.25);
}

// ---------------------------------------------------------------------------
// C. calculateOutputs
// ---------------------------------------------------------------------------

TEST(CalculateOutputs_ComputesWeightedSumPlusBias)
{
    // W = [[1,2,3],[4,5,6]], b = [0.5,-0.5], x = [2,1,0.5]
    // z0 = 2 + 2 + 1.5 + 0.5 = 6.0 ; z1 = 8 + 5 + 3 - 0.5 = 15.5
    core::Layer layer = makeLayer(2, 3, act::relu, act::reluDerivative);
    setRow(layer, 0, {1.0, 2.0, 3.0});
    setRow(layer, 1, {4.0, 5.0, 6.0});
    layer.neuronBiases = {0.5, -0.5};

    const double input[3] = {2.0, 1.0, 0.5};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], 6.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], 15.5, kTol);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 6.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 15.5, kTol);
}

TEST(CalculateOutputs_ReluZerosNegativePreActivationsButKeepsZ)
{
    // Same weights, x = [-1,-1,-1] -> z0 = -6 + 0.5 = -5.5 ; z1 = -15 - 0.5 = -15.5
    core::Layer layer = makeLayer(2, 3, act::relu, act::reluDerivative);
    setRow(layer, 0, {1.0, 2.0, 3.0});
    setRow(layer, 1, {4.0, 5.0, 6.0});
    layer.neuronBiases = {0.5, -0.5};

    const double input[3] = {-1.0, -1.0, -1.0};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], -5.5, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], -15.5, kTol);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.0, kTol);
}

TEST(CalculateOutputs_SigmoidMatchesReference)
{
    // W = [[1],[-1]], b = 0, x = [0.5] -> z = [0.5,-0.5]
    core::Layer layer = makeLayer(2, 1, act::sigmoid, act::sigmoidDerivative);
    setRow(layer, 0, {1.0});
    setRow(layer, 1, {-1.0});

    const double input[1] = {0.5};
    layer.calculateOutputs(input);

    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.6224593312018546, 1e-12);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.3775406687981454, 1e-12);
}

TEST(CalculateOutputs_SoftmaxNormalizesToOne)
{
    // weights are all zero, so z = b = [1,2,3]
    core::Layer layer = makeLayer(3, 2, act::softmax, act::sigmoidDerivative);
    layer.neuronBiases = {1.0, 2.0, 3.0};

    const double input[2] = {0.3, -0.7};
    layer.calculateOutputs(input);

    double sum = layer.neuronOutputsActivated[0] + layer.neuronOutputsActivated[1]
               + layer.neuronOutputsActivated[2];
    ASSERT_NEAR(sum, 1.0, 1e-12);
    ASSERT_NEAR_AT(0, layer.neuronOutputsActivated[0], 0.0900305731703805, 1e-12);
    ASSERT_NEAR_AT(1, layer.neuronOutputsActivated[1], 0.2447284710547977, 1e-12);
    ASSERT_NEAR_AT(2, layer.neuronOutputsActivated[2], 0.6652409557748218, 1e-12);
}

TEST(CalculateOutputs_SingleNeuronWithWideInput)
{
    // Exercises row indexing when layerWidth < inputSize.
    // w = [0,0.5,1,...,3.5], x = 1 -> z = 0.5*(0+1+..+7) = 14, + 0.25 bias
    core::Layer layer = makeLayer(1, 8, act::relu, act::reluDerivative);
    for (int i = 0; i < 8; ++i)
        layer.neuronWeights[i] = 0.5 * i;
    layer.neuronBiases[0] = 0.25;

    std::vector<double> input(8, 1.0);
    layer.calculateOutputs(input.data());

    ASSERT_NEAR(layer.neuronOutputsUnactivated[0], 14.25, kTol);
}

TEST(CalculateOutputs_IsIdempotentAcrossCalls)
{
    // A second call must overwrite, not accumulate into, the output buffers.
    core::Layer layer = makeLayer(2, 3, act::relu, act::reluDerivative);
    setRow(layer, 0, {1.0, 2.0, 3.0});
    setRow(layer, 1, {4.0, 5.0, 6.0});
    layer.neuronBiases = {0.5, -0.5};

    const double first[3] = {2.0, 1.0, 0.5};
    const double second[3] = {1.0, 0.0, 0.0};
    layer.calculateOutputs(first);
    layer.calculateOutputs(second);

    // z0 = 1 + 0.5 = 1.5 ; z1 = 4 - 0.5 = 3.5
    ASSERT_NEAR_AT(0, layer.neuronOutputsUnactivated[0], 1.5, kTol);
    ASSERT_NEAR_AT(1, layer.neuronOutputsUnactivated[1], 3.5, kTol);
}

// ---------------------------------------------------------------------------
// D. computeGradients
// ---------------------------------------------------------------------------

TEST(ComputeGradients_BiasGradientEqualsDelta)
{
    core::Layer layer = makeLayer(3, 2);
    ensureGradientBuffers(layer);
    layer.neuronDeltas = {0.5, -1.5, 2.0};

    layer.computeGradients();

    ASSERT_NEAR_AT(0, layer.neuronBiasGradients[0], 0.5, kTol);
    ASSERT_NEAR_AT(1, layer.neuronBiasGradients[1], -1.5, kTol);
    ASSERT_NEAR_AT(2, layer.neuronBiasGradients[2], 2.0, kTol);
}

TEST(ComputeGradients_WeightGradientEqualsDeltaTimesLayerInput)
{
    // dL/dW[n][i] = delta[n] * input[i], where input is the vector fed to
    // calculateOutputs (i.e. the previous layer's activations).
    core::Layer layer = makeLayer(2, 3, act::relu, act::reluDerivative);
    setRow(layer, 0, {1.0, 0.0, 0.0});
    setRow(layer, 1, {0.0, 1.0, 0.0});
    ensureGradientBuffers(layer);

    const double input[3] = {1.0, 2.0, 3.0};
    layer.calculateOutputs(input);
    layer.neuronDeltas = {0.5, -1.5};

    layer.computeGradients();

    for (int neuron = 0; neuron < layer.layerWidth; ++neuron)
        for (int i = 0; i < layer.inputSize; ++i)
        {
            double expected = layer.neuronDeltas[neuron] * input[i];
            double actual = layer.neuronWeightGradients[neuron * layer.inputSize + i];
            if (!testing::nearlyEqual(actual, expected, kTol))
                FAIL_WITH_NOTE(
                    "computeGradients() multiplies delta by this layer's own activations "
                    "(neuronOutputsActivated) instead of its input; the layer needs to keep "
                    "a pointer/copy of the input passed to calculateOutputs",
                    "weightGradient[%d][%d]: actual %.12g, expected delta[%d]*input[%d] = %.12g",
                    neuron, i, actual, neuron, i, expected);
        }
}

TEST(ComputeGradients_WideInputStaysInsideItsBuffers)
{
    // layerWidth(1) < inputSize(6): indexing activations by input index reads
    // 6 values out of a 1-element vector. ASan reports this as a heap overflow.
    core::Layer layer = makeLayer(1, 6, act::relu, act::reluDerivative);
    ensureGradientBuffers(layer);

    std::vector<double> input(6);
    for (int i = 0; i < 6; ++i) input[i] = 1.0 + i;
    layer.calculateOutputs(input.data());
    layer.neuronDeltas[0] = 2.0;

    layer.computeGradients();

    for (int i = 0; i < 6; ++i)
    {
        double expected = 2.0 * input[i];
        double actual = layer.neuronWeightGradients[i];
        if (!testing::nearlyEqual(actual, expected, kTol))
            FAIL_WITH_NOTE(
                "reads neuronOutputsActivated[input] where input runs to inputSize-1, "
                "but that vector only holds layerWidth values",
                "weightGradient[0][%d]: actual %.12g, expected %.12g", i, actual, expected);
    }
}

TEST(ComputeGradients_OnFreshLayerIsMemorySafe)
{
    // Reproduces the crash seen from main.cpp: runEpoch -> backwardPass ->
    // computeGradients writes into gradient vectors the constructor never sized.
    core::Layer layer = makeLayer(2, 2);
    const double input[2] = {0.1, 0.2};
    layer.calculateOutputs(input);
    layer.neuronDeltas = {0.5, 0.5};

    layer.computeGradients();

    if (layer.neuronWeightGradients.size() < 4 || layer.neuronBiasGradients.size() < 2)
        FAIL_WITH_NOTE(
            "computeGradients() wrote past the end of empty gradient vectors",
            "after the call neuronWeightGradients.size()=%zu neuronBiasGradients.size()=%zu",
            layer.neuronWeightGradients.size(), layer.neuronBiasGradients.size());
}

// ---------------------------------------------------------------------------
// E. updateWeightsAndBiases
// ---------------------------------------------------------------------------

TEST(Update_AppliesGradientDescentStep)
{
    core::Layer layer = makeLayer(2, 3);
    ensureGradientBuffers(layer);
    std::fill(layer.neuronWeights.begin(), layer.neuronWeights.end(), 1.0);
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 0.5);
    std::fill(layer.neuronBiases.begin(), layer.neuronBiases.end(), 2.0);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 1.0);

    layer.updateWeightsAndBiases(0.1);

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, layer.neuronWeights[i], 0.95, kTol);
    for (int i = 0; i < layer.layerWidth; ++i)
        ASSERT_NEAR_AT(i, layer.neuronBiases[i], 1.9, kTol);
}

TEST(Update_ZeroLearningRateIsNoOp)
{
    core::Layer layer = makeLayer(2, 2);
    ensureGradientBuffers(layer);
    setRow(layer, 0, {0.25, -0.5});
    setRow(layer, 1, {1.5, 2.0});
    layer.neuronBiases = {0.75, -1.25};
    std::fill(layer.neuronWeightGradients.begin(), layer.neuronWeightGradients.end(), 99.0);
    std::fill(layer.neuronBiasGradients.begin(), layer.neuronBiasGradients.end(), 99.0);

    layer.updateWeightsAndBiases(0.0);

    ASSERT_NEAR_AT(0, layer.neuronWeights[0], 0.25, kTol);
    ASSERT_NEAR_AT(1, layer.neuronWeights[1], -0.5, kTol);
    ASSERT_NEAR_AT(2, layer.neuronWeights[2], 1.5, kTol);
    ASSERT_NEAR_AT(3, layer.neuronWeights[3], 2.0, kTol);
    ASSERT_NEAR_AT(0, layer.neuronBiases[0], 0.75, kTol);
    ASSERT_NEAR_AT(1, layer.neuronBiases[1], -1.25, kTol);
}

TEST(Update_OnFreshLayerIsMemorySafe)
{
    // Same missing-allocation defect as ComputeGradients_OnFreshLayerIsMemorySafe,
    // but on the read side: indexes gradient vectors that are still empty.
    core::Layer layer = makeLayer(2, 2);
    layer.updateWeightsAndBiases(0.05);

    for (size_t i = 0; i < layer.neuronWeights.size(); ++i)
        ASSERT_FINITE(layer.neuronWeights[i]);
}

// ---------------------------------------------------------------------------
// F. OutputLayer::computeErrorSignal
// ---------------------------------------------------------------------------

TEST(OutputLayer_ErrorSignal_MseWithSigmoid)
{
    // z = [0,0] -> a = [0.5,0.5], target = [1,0], N = 2
    // dL/da = 2(a-t)/N = [-0.5, 0.5] ; sigmoid'(0) = 0.25
    // delta = [-0.125, 0.125]
    core::OutputLayer layer = makeOutputLayer(2, 2, act::sigmoid, act::sigmoidDerivative,
                                              loss::mse, loss::mseDerivative);
    layer.neuronOutputsUnactivated = {0.0, 0.0};
    layer.neuronOutputsActivated = {0.5, 0.5};

    const double target[2] = {1.0, 0.0};
    layer.computeErrorSignal(target);

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], -0.125, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 0.125, kTol);
}

TEST(OutputLayer_ErrorSignal_ReluGateBlocksNegativePreActivation)
{
    // z = [-1,2] -> a = [0,2], target = [1,0], N = 2
    // dL/da = [-1, 2] ; relu'(z) = [0,1] -> delta = [0, 2]
    core::OutputLayer layer = makeOutputLayer(2, 2, act::relu, act::reluDerivative,
                                              loss::mse, loss::mseDerivative);
    layer.neuronOutputsUnactivated = {-1.0, 2.0};
    layer.neuronOutputsActivated = {0.0, 2.0};

    const double target[2] = {1.0, 0.0};
    layer.computeErrorSignal(target);

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 2.0, kTol);
}

TEST(OutputLayer_ErrorSignal_PerfectPredictionGivesZeroDelta)
{
    core::OutputLayer layer = makeOutputLayer(3, 2, act::sigmoid, act::sigmoidDerivative,
                                              loss::mse, loss::mseDerivative);
    layer.neuronOutputsUnactivated = {0.4, -0.2, 1.1};
    layer.neuronOutputsActivated = {0.25, 0.5, 0.75};

    const double target[3] = {0.25, 0.5, 0.75};
    layer.computeErrorSignal(target);

    for (int i = 0; i < 3; ++i)
        ASSERT_NEAR_AT(i, layer.neuronDeltas[i], 0.0, kTol);
}

TEST(OutputLayer_ErrorSignal_BinaryCrossEntropyWithSigmoid)
{
    // z = [0] -> a = [0.5], target = [1], N = 1
    // dL/da = (a-t)/(a(1-a))/N = -0.5/0.25 = -2 ; sigmoid'(0) = 0.25
    // delta = -0.5   (== a - t, the familiar BCE+sigmoid simplification)
    core::OutputLayer layer = makeOutputLayer(1, 2, act::sigmoid, act::sigmoidDerivative,
                                              loss::binaryCrossEntropy,
                                              loss::binaryCrossEntropyDerivative);
    layer.neuronOutputsUnactivated = {0.0};
    layer.neuronOutputsActivated = {0.5};

    const double target[1] = {1.0};
    layer.computeErrorSignal(target);

    ASSERT_NEAR(layer.neuronDeltas[0], -0.5, kTol);
}

TEST(OutputLayer_ErrorSignal_UsesDistinctScratchPads)
{
    // Loss derivative and activation derivative must not share a buffer:
    // width 3 with inputSize 1 keeps both pads exactly layerWidth long.
    core::OutputLayer layer = makeOutputLayer(3, 1, act::sigmoid, act::sigmoidDerivative,
                                              loss::mse, loss::mseDerivative);
    layer.neuronOutputsUnactivated = {0.0, 0.0, 0.0};
    layer.neuronOutputsActivated = {0.5, 0.5, 0.5};

    const double target[3] = {1.0, 0.0, 1.0};
    layer.computeErrorSignal(target);

    // dL/da = 2(a-t)/3 = [-1/3, 1/3, -1/3] ; sigmoid'(0) = 0.25
    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], -1.0 / 12.0, 1e-12);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.0 / 12.0, 1e-12);
    ASSERT_NEAR_AT(2, layer.neuronDeltas[2], -1.0 / 12.0, 1e-12);
}

// ---------------------------------------------------------------------------
// G. HiddenLayer::computeErrorSignal
// ---------------------------------------------------------------------------

TEST(HiddenLayer_ErrorSignal_SingleNextNeuron)
{
    // this layer: width 2, inputSize 3, relu, z = [1,-1] -> relu' = [1,0]
    // next layer: width 1, inputSize 2, W = [0.5,-0.5], delta = [1]
    // expected: [1*0.5*1, 1*(-0.5)*0] = [0.5, 0]
    core::HiddenLayer layer = makeHiddenLayer(2, 3, act::relu, act::reluDerivative);
    layer.neuronOutputsUnactivated = {1.0, -1.0};

    std::vector<double> nextWeights = {0.5, -0.5};
    std::vector<double> nextDeltas = {1.0};

    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.5, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 0.0, kTol);
}

TEST(HiddenLayer_ErrorSignal_ReluGateZerosDeadNeuron)
{
    core::HiddenLayer layer = makeHiddenLayer(2, 2, act::relu, act::reluDerivative);
    layer.neuronOutputsUnactivated = {-1.0, 1.0};

    std::vector<double> nextWeights = {0.5, 0.5};
    std::vector<double> nextDeltas = {2.0};

    layer.computeErrorSignal(1, nextDeltas.data(), nextWeights.data());

    ASSERT_NEAR_AT(0, layer.neuronDeltas[0], 0.0, kTol);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.0, kTol);
}

TEST(HiddenLayer_ErrorSignal_AppliesDerivativeExactlyOnce)
{
    // width == inputSize == 2 so the weight stride is right either way; this
    // isolates the derivative multiply from the indexing.
    // next layer: width 2, inputSize 2, W = [[1,2],[3,4]], delta = [1,1]
    // sum0 = 1*1 + 1*3 = 4 ; sum1 = 1*2 + 1*4 = 6 ; sigmoid'(0) = 0.25
    // expected: [1.0, 1.5]
    core::HiddenLayer layer = makeHiddenLayer(2, 2, act::sigmoid, act::sigmoidDerivative);
    layer.neuronOutputsUnactivated = {0.0, 0.0};

    std::vector<double> nextWeights = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> nextDeltas = {1.0, 1.0};

    layer.computeErrorSignal(2, nextDeltas.data(), nextWeights.data());

    if (!testing::nearlyEqual(layer.neuronDeltas[0], 1.0, kTol))
        FAIL_WITH_NOTE(
            "the activation-derivative multiply sits inside the inner loop (missing braces), "
            "so it is applied once per next-layer neuron instead of once per neuron",
            "delta[0]: actual %.12g, expected 1.0", layer.neuronDeltas[0]);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 1.5, kTol);
}

TEST(HiddenLayer_ErrorSignal_StridesNextWeightsByOwnWidth)
{
    // width 2, inputSize 3: the next layer's row stride is ITS inputSize, which
    // equals THIS layer's layerWidth (2) -- not this layer's inputSize (3).
    // relu with z > 0 makes the derivative 1, isolating the indexing.
    // next layer: width 2, inputSize 2, W = [[1,2],[3,4]], delta = [1,1]
    // expected: [1+3, 2+4] = [4, 6]
    core::HiddenLayer layer = makeHiddenLayer(2, 3, act::relu, act::reluDerivative);
    layer.neuronOutputsUnactivated = {1.0, 1.0};

    std::vector<double> nextWeights = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> nextDeltas = {1.0, 1.0};

    layer.computeErrorSignal(2, nextDeltas.data(), nextWeights.data());

    if (!testing::nearlyEqual(layer.neuronDeltas[0], 4.0, kTol))
        FAIL_WITH_NOTE(
            "indexes nextLayerWeights with stride this->inputSize; the stride must be "
            "this->layerWidth, which also reads past the end of the next layer's weights",
            "delta[0]: actual %.12g, expected 4.0", layer.neuronDeltas[0]);
    ASSERT_NEAR_AT(1, layer.neuronDeltas[1], 6.0, kTol);
}

// ---------------------------------------------------------------------------
// H. multi-layer integration
// ---------------------------------------------------------------------------

TEST(Integration_TwoLayerForwardChain)
{
    // hidden: width 3, inputSize 2, relu, W = [[1,0],[0,1],[1,1]], b = 0
    //   x = [0.5,0.25] -> z = a = [0.5, 0.25, 0.75]
    // output: width 1, inputSize 3, sigmoid, W = [1,1,1], b = 0
    //   z = 1.5 -> a = sigmoid(1.5)
    core::HiddenLayer hidden = makeHiddenLayer(3, 2, act::relu, act::reluDerivative);
    setRow(hidden, 0, {1.0, 0.0});
    setRow(hidden, 1, {0.0, 1.0});
    setRow(hidden, 2, {1.0, 1.0});

    core::OutputLayer output = makeOutputLayer(1, 3, act::sigmoid, act::sigmoidDerivative,
                                               loss::mse, loss::mseDerivative);
    setRow(output, 0, {1.0, 1.0, 1.0});

    const double input[2] = {0.5, 0.25};
    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());

    ASSERT_NEAR_AT(0, hidden.neuronOutputsActivated[0], 0.5, kTol);
    ASSERT_NEAR_AT(1, hidden.neuronOutputsActivated[1], 0.25, kTol);
    ASSERT_NEAR_AT(2, hidden.neuronOutputsActivated[2], 0.75, kTol);
    ASSERT_NEAR(output.neuronOutputsUnactivated[0], 1.5, kTol);
    ASSERT_NEAR(output.neuronOutputsActivated[0], 0.8175744761936437, 1e-12);
}

TEST(Integration_ForwardBackwardUpdateLowersLoss)
{
    // One gradient step on a fixed (input, target) pair must not increase the
    // loss. Mirrors main.cpp's geometry: 2 -> hidden 2 -> output 1.
    core::HiddenLayer hidden = makeHiddenLayer(2, 2, act::sigmoid, act::sigmoidDerivative);
    core::OutputLayer output = makeOutputLayer(1, 2, act::sigmoid, act::sigmoidDerivative,
                                               loss::mse, loss::mseDerivative);
    setRow(hidden, 0, {0.5, -0.3});
    setRow(hidden, 1, {0.2, 0.8});
    setRow(output, 0, {0.7, -0.4});
    ensureGradientBuffers(hidden);
    ensureGradientBuffers(output);

    const double input[2] = {0.1, 0.2};
    const double target[1] = {1.0};

    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());
    double lossBefore = loss::mse(target, output.neuronOutputsActivated.data(), 1);

    output.computeErrorSignal(target);
    output.computeGradients();
    hidden.computeErrorSignal(output.layerWidth,
                              output.neuronDeltas.data(),
                              output.neuronWeights.data());
    hidden.computeGradients();
    hidden.updateWeightsAndBiases(0.5);
    output.updateWeightsAndBiases(0.5);

    hidden.calculateOutputs(input);
    output.calculateOutputs(hidden.neuronOutputsActivated.data());
    double lossAfter = loss::mse(target, output.neuronOutputsActivated.data(), 1);

    ASSERT_FINITE(lossAfter);
    if (lossAfter > lossBefore + 1e-12)
        FAIL_WITH_NOTE(
            "a single descent step moved uphill, so at least one gradient has the wrong "
            "value or sign",
            "loss before %.12g, loss after %.12g", lossBefore, lossAfter);
}

TEST(Print_DoesNotCrashOnInitializedLayer)
{
    core::Layer layer = makeLayer(2, 3, act::relu, act::reluDerivative, init::xavier);
    layer.initializeWeightsAndBiases();
    // print() writes to stdout; we only care that it stays in bounds.
    std::freopen("/dev/null", "w", stdout);
    layer.print();
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    return testing::runAll("core/layer.h", argc > 1 ? argv[1] : nullptr);
}
