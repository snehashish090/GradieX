//
// Unit tests for core/network.h : Network.
//
// Companion to test_layer.cpp. Same harness, same rule: expected values are
// computed in the test from the definitions (sigmoid, MSE, the gradient-descent
// update), never captured from the implementation's own output.
//
// The layers are exercised here only as far as the Network drives them;
// per-function layer behaviour lives in test_layer.cpp.
//
// 0.0.3 replaced the per-sample backwardPass() with accumulate-then-apply:
// zeroAllGradients() / accumulateGradients(target) / scaleAllGradients(1/n) /
// applyGradients(). One SGD step on a single sample is that sequence with n=1,
// which is what sgdStep() below does.
//

#include "test_framework.h"
#include "../core/network.h"

#include <vector>

namespace act = core::functions::activations;
namespace init = core::functions::initializers;
namespace loss = core::functions::loss;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// float32 engine: reference values are computed in double, so allow the ~1e-7
// resolution of a float32 result near 1.0.
static constexpr double kTol = 1e-6;
static constexpr double kExact = 0.0;

// Reference sigmoid / derivative, straight from the definition.
static double sig(double z) { return 1.0 / (1.0 + std::exp(-z)); }
static double sigPrime(double z) { double s = sig(z); return s * (1.0 - s); }

// train(verbose=true) writes to stdout; tests that ask for it only care about
// the numbers.
static void silenceStdout() { std::freopen("/dev/null", "w", stdout); }

static void addSigmoidHidden(core::Network& net, int index, int width, int inputSize)
{
    net.addHiddenLayer(index, width, inputSize, act::sigmoid, init::zeros, loss::mse);
}

static void addSigmoidOutput(core::Network& net, int index, int width, int inputSize)
{
    net.addOutputLayer(index, width, inputSize, act::sigmoid, init::zeros, loss::mse);
}

static void setRow(core::Layer& layer, int neuron, std::vector<float> row)
{
    for (int i = 0; i < static_cast<int>(row.size()); ++i)
        layer.neuronWeights[neuron * layer.inputSize + i] = row[i];
}

// One gradient-descent step on a single sample, spelled out with the engine's
// own primitives. trainBatch() with batchSize 1 does exactly this (its
// scaleAllGradients(1.0f) is an exact no-op).
static void sgdStep(core::Network& net, const std::vector<float>& input,
                    std::vector<float>& target)
{
    net.zeroAllGradients();
    net.forwardPass(input.data());
    net.accumulateGradients(target);
    net.applyGradients();
}

// The fixture every forward/backward test below shares:
//
//   x = [0.1, 0.2]              target = [1.0]        learningRate = 0.5
//   hidden (2x2, sigmoid)  W = [[0.5,-0.3],[0.2,0.8]]  b = 0
//   output (1x2, sigmoid)  W = [[0.7,-0.4]]            b = 0
//
struct Fixture
{
    core::Network net{1, 1, 0.5f};
    std::vector<float> input{0.1f, 0.2f};
    std::vector<float> target{1.0f};

    // hand-derived reference values for that geometry
    double zh0, zh1, ah0, ah1, zo, ao;

    Fixture()
    {
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});

        zh0 = 0.5 * 0.1 + (-0.3) * 0.2;
        zh1 = 0.2 * 0.1 + 0.8 * 0.2;
        ah0 = sig(zh0);
        ah1 = sig(zh1);
        zo = 0.7 * ah0 + (-0.4) * ah1;
        ao = sig(zo);
    }

    // delta for the single output neuron: dL/da * da/dz, N = 1
    double outputDelta() const { return 2.0 * (ao - 1.0) * sigPrime(zo); }

    // delta for hidden neuron j: (outDelta * outW[0][j]) * sigma'(zh_j)
    double hiddenDelta(int j) const
    {
        const double outW[2] = {0.7, -0.4};
        const double zh[2] = {zh0, zh1};
        return outputDelta() * outW[j] * sigPrime(zh[j]);
    }
};

// ---------------------------------------------------------------------------
// A. construction and layer registration
// ---------------------------------------------------------------------------

TEST(Construction_SizesLayerArrays)
{
    core::Network net(3, 2, 0.05f);
    ASSERT_EQ_INT(net.numHiddenLayers, 3);
    ASSERT_EQ_INT(net.numOutputLayers, 2);
    ASSERT_NEAR(net.learningRate, 0.05, kTol);
    ASSERT_EQ_INT(net.batches, 1);
    ASSERT_EQ_INT(net.hiddenLayers.size(), 3);
    ASSERT_EQ_INT(net.outputLayers.size(), 2);
    // resize() value-initialises the pointers, so the destructor is safe on
    // slots that were never filled.
    for (auto* layer : net.hiddenLayers) ASSERT_TRUE(layer == nullptr);
    for (auto* layer : net.outputLayers) ASSERT_TRUE(layer == nullptr);
}

TEST(Construction_StoresBatchSize)
{
    core::Network net(1, 1, 0.05f, 16);
    ASSERT_EQ_INT(net.batches, 16);
}

TEST(DefaultConstructor_LayerCountsMatchAllocation)
{
    // A default-constructed network owns nothing, so the counts must read 0.
    // Anything that trusts them -- accumulateGradients() indexes outputLayers[0]
    // unconditionally -- would otherwise read past the end of an empty vector.
    core::Network net;
    if (net.outputLayers.size() != static_cast<size_t>(net.numOutputLayers)
        || net.hiddenLayers.size() != static_cast<size_t>(net.numHiddenLayers))
        FAIL_WITH_NOTE(
            "the default constructor must keep the counts and the vectors in step "
            "(resize them, or report 0 output layers)",
            "numHiddenLayers=%d hiddenLayers.size()=%zu | numOutputLayers=%d outputLayers.size()=%zu",
            net.numHiddenLayers, net.hiddenLayers.size(),
            net.numOutputLayers, net.outputLayers.size());
}

TEST(AddHiddenLayer_StoresLayerWithRequestedGeometry)
{
    core::Network net(2, 1, 0.01f);
    addSigmoidHidden(net, 0, 4, 2);
    addSigmoidHidden(net, 1, 3, 4);

    ASSERT_TRUE(net.hiddenLayers[0] != nullptr);
    ASSERT_EQ_INT(net.hiddenLayers[0]->layerWidth, 4);
    ASSERT_EQ_INT(net.hiddenLayers[0]->inputSize, 2);
    ASSERT_EQ_INT(net.hiddenLayers[1]->layerWidth, 3);
    ASSERT_EQ_INT(net.hiddenLayers[1]->inputSize, 4);
    ASSERT_TRUE(net.hiddenLayers[0]->activationFunction == act::sigmoid);
    ASSERT_TRUE(net.hiddenLayers[0]->activationFunctionDerivative == act::sigmoidDerivative);
}

TEST(AddOutputLayer_StoresLayerWithRequestedGeometry)
{
    core::Network net(0, 2, 0.01f);
    addSigmoidOutput(net, 0, 1, 3);
    addSigmoidOutput(net, 1, 5, 3);

    ASSERT_EQ_INT(net.outputLayers[0]->layerWidth, 1);
    ASSERT_EQ_INT(net.outputLayers[0]->inputSize, 3);
    ASSERT_EQ_INT(net.outputLayers[1]->layerWidth, 5);
    ASSERT_TRUE(net.outputLayers[0]->lossFunction == loss::mse);
    ASSERT_TRUE(net.outputLayers[0]->lossFunctionDerivative == loss::mseDerivative);
}

TEST(AddHiddenLayer_OutOfRangeIndexIsRejected)
{
    // hiddenLayers[layerIndex] is an unchecked operator[] write, so an index
    // past the end would scribble over the heap (ASan: heap overflow) and leak
    // the layer it just allocated. The index must be checked instead.
    core::Network net(1, 1, 0.01f);
    bool rejected = false;
    try
    {
        addSigmoidHidden(net, 2, 4, 2);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }

    if (!rejected)
        FAIL_WITH_NOTE(
            "addHiddenLayer/addOutputLayer must bounds-check layerIndex (throw, or use .at()) "
            "instead of writing past the end of the layer vector",
            "addHiddenLayer(index=2) on a network with %zu hidden slots returned normally",
            net.hiddenLayers.size());
}

TEST(AddOutputLayer_OutOfRangeIndexIsRejected)
{
    core::Network net(1, 1, 0.01f);
    bool rejected = false;
    try { addSigmoidOutput(net, 3, 1, 2); }
    catch (const std::exception&) { rejected = true; }

    if (!rejected)
        FAIL_WITH_NOTE("addOutputLayer must bounds-check layerIndex",
                       "addOutputLayer(index=3) on a network with %zu output slots returned normally",
                       net.outputLayers.size());
}

TEST(AddHiddenLayer_ReplacingAFilledSlotDoesNotLeak)
{
    // ASan/LSan is the real assertion: overwriting a slot must delete the layer
    // that was there.
    core::Network net(1, 1, 0.01f);
    addSigmoidHidden(net, 0, 4, 2);
    addSigmoidHidden(net, 0, 6, 2);

    ASSERT_EQ_INT(net.hiddenLayers[0]->layerWidth, 6);
}

TEST(InitializeAllWeightsAndBiases_ReachesEveryLayer)
{
    core::Network net(2, 1, 0.01f);
    net.addHiddenLayer(0, 4, 2, act::sigmoid, init::xavier, loss::mse);
    net.addHiddenLayer(1, 3, 4, act::sigmoid, init::xavier, loss::mse);
    net.addOutputLayer(0, 1, 3, act::sigmoid, init::xavier, loss::mse);

    const float sentinel = -98765.0f;
    for (auto* layer : net.hiddenLayers)
        std::fill(layer->neuronWeights.begin(), layer->neuronWeights.end(), sentinel);
    std::fill(net.outputLayers[0]->neuronWeights.begin(),
              net.outputLayers[0]->neuronWeights.end(), sentinel);

    net.initializeAllWeightsAndBiases();

    for (auto* layer : net.hiddenLayers)
        for (size_t i = 0; i < layer->neuronWeights.size(); ++i)
        {
            ASSERT_TRUE(layer->neuronWeights[i] != sentinel);
            ASSERT_FINITE(layer->neuronWeights[i]);
        }
    for (size_t i = 0; i < net.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_TRUE(net.outputLayers[0]->neuronWeights[i] != sentinel);
}

// ---------------------------------------------------------------------------
// B. forwardPass
// ---------------------------------------------------------------------------

TEST(ForwardPass_ChainsHiddenActivationsIntoOutputLayer)
{
    Fixture f;
    f.net.forwardPass(f.input.data());

    core::HiddenLayer& hidden = *f.net.hiddenLayers[0];
    core::OutputLayer& output = *f.net.outputLayers[0];

    ASSERT_NEAR_AT(0, hidden.neuronOutputsUnactivated[0], f.zh0, kTol);
    ASSERT_NEAR_AT(1, hidden.neuronOutputsUnactivated[1], f.zh1, kTol);
    ASSERT_NEAR_AT(0, hidden.neuronOutputsActivated[0], f.ah0, kTol);
    ASSERT_NEAR_AT(1, hidden.neuronOutputsActivated[1], f.ah1, kTol);
    ASSERT_NEAR(output.neuronOutputsUnactivated[0], f.zo, kTol);
    ASSERT_NEAR(output.neuronOutputsActivated[0], f.ao, kTol);
}

TEST(ForwardPass_RecordsEachLayersOwnInput)
{
    // computeGradients() reads layerInputs, so forwardPass must leave the
    // network's input in the hidden layer and the hidden activations in the
    // output layer.
    Fixture f;
    f.net.forwardPass(f.input.data());

    core::HiddenLayer& hidden = *f.net.hiddenLayers[0];
    core::OutputLayer& output = *f.net.outputLayers[0];

    ASSERT_EQ_INT(hidden.layerInputs.size(), 2);
    ASSERT_NEAR_AT(0, hidden.layerInputs[0], 0.1, kTol);
    ASSERT_NEAR_AT(1, hidden.layerInputs[1], 0.2, kTol);
    ASSERT_NEAR_AT(0, output.layerInputs[0], f.ah0, kTol);
    ASSERT_NEAR_AT(1, output.layerInputs[1], f.ah1, kTol);
}

TEST(ForwardPass_IsRepeatableForTheSameInput)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    float first = f.net.outputLayers[0]->neuronOutputsActivated[0];

    std::vector<float> other{0.9f, 0.4f};
    f.net.forwardPass(other.data());
    f.net.forwardPass(f.input.data());
    float again = f.net.outputLayers[0]->neuronOutputsActivated[0];

    ASSERT_NEAR(again, first, kExact);
}

// ---------------------------------------------------------------------------
// C. accumulateGradients / applyGradients
// ---------------------------------------------------------------------------

TEST(AccumulateGradients_SetsOutputDeltaFromTarget)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    f.net.zeroAllGradients();
    f.net.accumulateGradients(f.target);

    ASSERT_NEAR(f.net.outputLayers[0]->neuronDeltas[0], f.outputDelta(), kTol);
}

TEST(AccumulateGradients_PropagatesDeltaIntoHiddenLayer)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    f.net.zeroAllGradients();
    f.net.accumulateGradients(f.target);

    core::HiddenLayer& hidden = *f.net.hiddenLayers[0];
    ASSERT_NEAR_AT(0, hidden.neuronDeltas[0], f.hiddenDelta(0), kTol);
    ASSERT_NEAR_AT(1, hidden.neuronDeltas[1], f.hiddenDelta(1), kTol);
}

TEST(AccumulateGradients_DoesNotUpdateWeightsOnItsOwn)
{
    // The split between accumulate and apply is what makes mini-batching
    // possible; accumulating must leave the parameters alone.
    Fixture f;
    std::vector<float> outBefore = f.net.outputLayers[0]->neuronWeights;
    std::vector<float> hidBefore = f.net.hiddenLayers[0]->neuronWeights;

    f.net.forwardPass(f.input.data());
    f.net.zeroAllGradients();
    f.net.accumulateGradients(f.target);

    for (size_t i = 0; i < outBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.outputLayers[0]->neuronWeights[i], outBefore[i], kExact);
    for (size_t i = 0; i < hidBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.hiddenLayers[0]->neuronWeights[i], hidBefore[i], kExact);
}

TEST(ApplyGradients_AppliesOneGradientStepToEveryWeight)
{
    Fixture f;
    const double lr = 0.5;
    const double outWBefore[2] = {0.7, -0.4};
    const double hiddenWBefore[2][2] = {{0.5, -0.3}, {0.2, 0.8}};

    sgdStep(f.net, f.input, f.target);

    core::HiddenLayer& hidden = *f.net.hiddenLayers[0];
    core::OutputLayer& output = *f.net.outputLayers[0];

    // output weights: W -= lr * delta * (hidden activation)
    const double ah[2] = {f.ah0, f.ah1};
    for (int i = 0; i < 2; ++i)
        ASSERT_NEAR_AT(i, output.neuronWeights[i],
                       outWBefore[i] - lr * f.outputDelta() * ah[i], kTol);
    ASSERT_NEAR(output.neuronBiases[0], 0.0 - lr * f.outputDelta(), kTol);

    // hidden weights: W -= lr * delta * (network input)
    for (int neuron = 0; neuron < 2; ++neuron)
    {
        for (int i = 0; i < 2; ++i)
            ASSERT_NEAR_AT(i, hidden.neuronWeights[neuron * 2 + i],
                           hiddenWBefore[neuron][i] - lr * f.hiddenDelta(neuron) * f.input[i],
                           kTol);
        ASSERT_NEAR_AT(neuron, hidden.neuronBiases[neuron],
                       0.0 - lr * f.hiddenDelta(neuron), kTol);
    }
}

TEST(AccumulateGradients_SumsOverAMiniBatch)
{
    // Two samples accumulated without an intervening apply must leave the sum
    // of the two per-sample bias gradients on the output neuron.
    Fixture f;

    f.net.zeroAllGradients();
    f.net.forwardPass(f.input.data());
    f.net.accumulateGradients(f.target);
    double firstDelta = f.net.outputLayers[0]->neuronDeltas[0];
    double afterFirst = f.net.outputLayers[0]->neuronBiasGradients[0];

    std::vector<float> other{0.9f, 0.4f};
    f.net.forwardPass(other.data());
    f.net.accumulateGradients(f.target);
    double secondDelta = f.net.outputLayers[0]->neuronDeltas[0];
    double afterSecond = f.net.outputLayers[0]->neuronBiasGradients[0];

    ASSERT_NEAR(afterFirst, firstDelta, kTol);
    ASSERT_NEAR(afterSecond, firstDelta + secondDelta, kTol);
}

TEST(ZeroAllGradients_ClearsEveryLayer)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    f.net.accumulateGradients(f.target);
    f.net.zeroAllGradients();

    for (float g : f.net.outputLayers[0]->neuronWeightGradients)
        ASSERT_NEAR(g, 0.0, kExact);
    for (float g : f.net.hiddenLayers[0]->neuronWeightGradients)
        ASSERT_NEAR(g, 0.0, kExact);
    for (float g : f.net.hiddenLayers[0]->neuronBiasGradients)
        ASSERT_NEAR(g, 0.0, kExact);
}

TEST(ScaleAllGradients_ReachesEveryLayer)
{
    Fixture f;
    f.net.zeroAllGradients();
    f.net.forwardPass(f.input.data());
    f.net.accumulateGradients(f.target);

    std::vector<float> outBefore = f.net.outputLayers[0]->neuronWeightGradients;
    std::vector<float> hidBefore = f.net.hiddenLayers[0]->neuronWeightGradients;

    f.net.scaleAllGradients(0.5f);

    for (size_t i = 0; i < outBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.outputLayers[0]->neuronWeightGradients[i],
                       0.5 * outBefore[i], kTol);
    for (size_t i = 0; i < hidBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.hiddenLayers[0]->neuronWeightGradients[i],
                       0.5 * hidBefore[i], kTol);
}

TEST(RepeatedStepsConvergeOnOneSample)
{
    // 200 steps on a single (input, target) pair must drive the loss down; this
    // is the end-to-end check that every sign in the chain rule is right.
    Fixture f;
    f.net.forwardPass(f.input.data());
    double lossBefore = f.net.computeLoss(f.target);

    for (int step = 0; step < 200; ++step)
        sgdStep(f.net, f.input, f.target);

    f.net.forwardPass(f.input.data());
    double lossAfter = f.net.computeLoss(f.target);

    ASSERT_FINITE(lossAfter);
    if (!(lossAfter < lossBefore))
        FAIL_WITH_NOTE(
            "200 descent steps on a single sample did not reduce the loss, so a gradient "
            "somewhere in the chain has the wrong sign or magnitude",
            "loss before %.12g, loss after %.12g", lossBefore, lossAfter);
}

// ---------------------------------------------------------------------------
// D. computeLoss
// ---------------------------------------------------------------------------

TEST(ComputeLoss_IsZeroWhenPredictionMatchesTarget)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    std::vector<float> selfTarget{f.net.outputLayers[0]->neuronOutputsActivated[0]};

    ASSERT_NEAR(f.net.computeLoss(selfTarget), 0.0, kExact);
}

TEST(ComputeLoss_IsPositiveAndGrowsWithError)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    std::vector<float> near{static_cast<float>(f.ao) + 0.1f};
    std::vector<float> far{static_cast<float>(f.ao) + 0.4f};

    double lossNear = f.net.computeLoss(near);
    double lossFar = f.net.computeLoss(far);
    ASSERT_TRUE(lossNear > 0.0);
    ASSERT_TRUE(lossFar > lossNear);
}

TEST(ComputeLoss_AgreesWithTheConfiguredLossFunction)
{
    // The number reported during training must be the quantity being minimised,
    // so computeLoss() calls the layer's own lossFunction rather than hardcoding
    // a squared-error sum. With mse and a 3-wide output the two differ by 1/N.
    core::Network net(0, 1, 0.01f);
    addSigmoidOutput(net, 0, 3, 2);
    net.outputLayers[0]->neuronOutputsActivated = {0.5f, 0.25f, 0.75f};

    std::vector<float> target{1.0f, 0.0f, 0.0f};
    double reported = net.computeLoss(target);
    double configured = loss::mse(target.data(),
                                  net.outputLayers[0]->neuronOutputsActivated.data(), 3);

    if (!testing::nearlyEqual(reported, configured, kTol))
        FAIL_WITH_NOTE(
            "computeLoss() must call the layer's lossFunction rather than hardcoding "
            "a squared-error sum, otherwise the reported loss is not the quantity being minimised",
            "computeLoss() = %.12g, layer's mse() = %.12g", reported, configured);
}

TEST(ComputeLoss_FollowsANonMseLossFunction)
{
    core::Network net(0, 1, 0.01f);
    net.addOutputLayer(0, 3, 2, act::sigmoid, init::zeros, loss::mae);
    net.outputLayers[0]->neuronOutputsActivated = {0.5f, 0.25f, 0.75f};

    std::vector<float> target{1.0f, 0.0f, 0.0f};
    // mae = (|0.5-1| + |0.25-0| + |0.75-0|) / 3 = 1.5/3 = 0.5
    ASSERT_NEAR(net.computeLoss(target), 0.5, kTol);
}

// ---------------------------------------------------------------------------
// E. trainBatch
// ---------------------------------------------------------------------------

TEST(TrainBatch_AveragesGradientsOverTheBatch)
{
    // A batch of n must apply the MEAN of the per-sample gradients, so running
    // the same sample twice in one batch equals one single-sample step.
    std::vector<std::vector<float>> twice = {{0.1f, 0.2f}, {0.1f, 0.2f}};
    std::vector<std::vector<float>> targets = {{1.0f}, {1.0f}};

    Fixture batched;
    batched.net.trainBatch(0, 2, twice, targets);

    Fixture single;
    sgdStep(single.net, single.input, single.target);

    for (size_t i = 0; i < single.net.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, batched.net.outputLayers[0]->neuronWeights[i],
                       single.net.outputLayers[0]->neuronWeights[i], kTol);
    for (size_t i = 0; i < single.net.hiddenLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, batched.net.hiddenLayers[0]->neuronWeights[i],
                       single.net.hiddenLayers[0]->neuronWeights[i], kTol);
}

TEST(TrainBatch_ReturnsTheMeanBatchLoss)
{
    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.9f, 0.4f}};
    std::vector<std::vector<float>> targets = {{1.0f}, {0.0f}};

    Fixture f;
    // Losses are measured on the weights as they were at the start of the batch,
    // because the update only lands after every sample has been accumulated.
    f.net.forwardPass(inputs[0].data());
    double first = f.net.computeLoss(targets[0]);
    f.net.forwardPass(inputs[1].data());
    double second = f.net.computeLoss(targets[1]);

    Fixture g;
    double reported = g.net.trainBatch(0, 2, inputs, targets);

    ASSERT_NEAR(reported, (first + second) / 2.0, kTol);
}

TEST(TrainBatch_HonoursTheSampleOrderArray)
{
    // The order-aware overload is what lets train() shuffle. Visiting [1,0] must
    // match visiting a dataset that was written in that order.
    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.9f, 0.4f}};
    std::vector<std::vector<float>> targets = {{1.0f}, {0.0f}};
    std::vector<std::vector<float>> reversedIn = {inputs[1], inputs[0]};
    std::vector<std::vector<float>> reversedTg = {targets[1], targets[0]};

    const size_t order[2] = {1, 0};

    Fixture ordered;
    ordered.net.trainBatch(order, 0, 2, inputs, targets);

    Fixture rewritten;
    rewritten.net.trainBatch(nullptr, 0, 2, reversedIn, reversedTg);

    for (size_t i = 0; i < ordered.net.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, ordered.net.outputLayers[0]->neuronWeights[i],
                       rewritten.net.outputLayers[0]->neuronWeights[i], kExact);
}

// ---------------------------------------------------------------------------
// F. train
// ---------------------------------------------------------------------------

// Every train() call needs a loss buffer; this keeps the tests readable.
static std::vector<float> trainFor(core::Network& net, int epochs,
                                   std::vector<std::vector<float>>& inputs,
                                   std::vector<std::vector<float>>& targets,
                                   bool shuffle = false,
                                   std::optional<unsigned> seed = std::nullopt)
{
    std::vector<float> history(epochs > 0 ? static_cast<size_t>(epochs) : 0u);
    net.train(inputs, targets, epochs, history.data(), epochs, false, shuffle, seed);
    return history;
}

TEST(Train_MoreEpochsThanSamplesStaysInBounds)
{
    // Regression test: train() used to index the dataset by the epoch counter,
    // so it read past the end of the sample vector on epoch == numSamples.
    // 6 samples, 100 epochs.
    core::Network net(1, 1, 0.05f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);
    net.initializeAllWeightsAndBiases();

    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f}, {0.8f, 0.9f}, {0.2f, 0.1f}, {0.9f, 0.7f}, {0.4f, 0.5f}, {0.6f, 0.4f}};
    std::vector<std::vector<float>> targets = {
        {0.0f}, {1.0f}, {0.0f}, {1.0f}, {0.0f}, {1.0f}};

    trainFor(net, 100, inputs, targets);

    for (float w : net.outputLayers[0]->neuronWeights) ASSERT_FINITE(w);
    for (float w : net.hiddenLayers[0]->neuronWeights) ASSERT_FINITE(w);
}

TEST(Train_FillsOneLossPerEpoch)
{
    core::Network net(1, 1, 0.5f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);
    net.initializeAllWeightsAndBiases();

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}};

    std::vector<float> history(20, -1.0f);
    net.train(inputs, targets, 20, history.data(), 20);

    for (size_t i = 0; i < history.size(); ++i)
    {
        ASSERT_FINITE(history[i]);
        ASSERT_TRUE(history[i] >= 0.0f);   // every slot was written
    }
}

TEST(Train_RejectsALossBufferSmallerThanTheEpochCount)
{
    core::Network net(1, 1, 0.5f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}};
    std::vector<std::vector<float>> targets = {{1.0f}};
    std::vector<float> tooSmall(3);

    bool threw = false;
    try { net.train(inputs, targets, 10, tooSmall.data(), 3); }
    catch (const std::invalid_argument&) { threw = true; }

    if (!threw)
        FAIL_WITH_NOTE("train() must reject a loss buffer shorter than numEpochs rather than "
                       "writing past the end of it",
                       "train(numEpochs=10, bufferSize=3) returned normally");
}

TEST(Train_RejectsANullLossBuffer)
{
    core::Network net(1, 1, 0.5f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}};
    std::vector<std::vector<float>> targets = {{1.0f}};

    bool threw = false;
    try { net.train(inputs, targets, 5, nullptr, 5); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}

TEST(Train_OneEpochVisitsEverySampleInOrder)
{
    // Two identically-weighted networks: one trained for a single epoch with
    // batch size 1, the other stepped by hand over each sample in order. They
    // must end up identical.
    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}, {0.4f, 0.5f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {1.0f}};

    auto build = [](core::Network& net) {
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});
    };

    core::Network trained(1, 1, 0.5f, 1);
    core::Network stepped(1, 1, 0.5f, 1);
    build(trained);
    build(stepped);

    trainFor(trained, 1, inputs, targets);
    for (size_t sample = 0; sample < inputs.size(); ++sample)
        sgdStep(stepped, inputs[sample], targets[sample]);

    for (size_t i = 0; i < stepped.hiddenLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, trained.hiddenLayers[0]->neuronWeights[i],
                       stepped.hiddenLayers[0]->neuronWeights[i], kExact);
    for (size_t i = 0; i < stepped.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, trained.outputLayers[0]->neuronWeights[i],
                       stepped.outputLayers[0]->neuronWeights[i], kExact);
}

TEST(Train_BatchSizeZeroMeansOneBatchPerEpoch)
{
    // batches <= 0 falls back to the whole dataset, so one epoch is one update
    // and must equal a single hand-run full batch.
    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}, {0.4f, 0.5f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {1.0f}};

    core::Network wholeBatch(1, 1, 0.5f, 0);
    core::Network byHand(1, 1, 0.5f, 0);
    auto build = [](core::Network& net) {
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});
    };
    build(wholeBatch);
    build(byHand);

    trainFor(wholeBatch, 1, inputs, targets);
    byHand.trainBatch(0, 3, inputs, targets);

    for (size_t i = 0; i < byHand.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, wholeBatch.outputLayers[0]->neuronWeights[i],
                       byHand.outputLayers[0]->neuronWeights[i], kExact);
}

TEST(Train_ShufflingIsReproducibleForAGivenSeed)
{
    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f}, {0.8f, 0.9f}, {0.2f, 0.1f}, {0.9f, 0.7f}, {0.4f, 0.5f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {0.0f}, {1.0f}, {0.0f}};

    auto run = [&inputs, &targets](unsigned seed) {
        core::Network net(1, 1, 0.5f, 2);
        addSigmoidHidden(net, 0, 3, 2);
        addSigmoidOutput(net, 0, 1, 3);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.hiddenLayers[0], 2, {-0.6f, 0.4f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f, 0.2f});
        return trainFor(net, 30, inputs, targets, true, seed);
    };

    std::vector<float> a = run(99);
    std::vector<float> b = run(99);
    std::vector<float> c = run(100);

    for (size_t i = 0; i < a.size(); ++i)
        ASSERT_NEAR_AT(i, a[i], b[i], kExact);

    bool differs = false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != c[i]) { differs = true; break; }
    if (!differs)
        FAIL_WITH_NOTE("a different shuffle seed should produce a different visiting order",
                       "seeds 99 and 100 produced identical loss histories");
}

TEST(Train_ShuffleVisitsEverySampleOncePerEpoch)
{
    // A shuffle must permute the order, not resample it: one epoch with batch
    // size == the dataset accumulates every sample exactly once, so the update
    // is identical whether or not the order was shuffled.
    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f}, {0.8f, 0.9f}, {0.2f, 0.1f}, {0.9f, 0.7f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {0.0f}, {1.0f}};

    auto run = [&inputs, &targets](bool shuffle) {
        core::Network net(1, 1, 0.5f, 0);     // one batch per epoch
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});
        trainFor(net, 1, inputs, targets, shuffle, 7u);
        return net.outputLayers[0]->neuronBiases[0];
    };

    // Summation order differs, so allow float32 slack rather than demanding
    // bit equality.
    ASSERT_NEAR(run(true), run(false), kTol);
}

TEST(Train_LowersLossAcrossEpochs)
{
    core::Network net(1, 1, 0.5f);
    addSigmoidHidden(net, 0, 3, 2);
    addSigmoidOutput(net, 0, 1, 3);
    setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
    setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
    setRow(*net.hiddenLayers[0], 2, {-0.6f, 0.4f});
    setRow(*net.outputLayers[0], 0, {0.7f, -0.4f, 0.2f});

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}, {0.2f, 0.1f}, {0.9f, 0.7f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}, {0.0f}, {1.0f}};

    auto datasetLoss = [&net, &inputs, &targets]() {
        double total = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            net.forwardPass(inputs[i].data());
            total += net.computeLoss(targets[i]);
        }
        return total / static_cast<double>(inputs.size());
    };

    double before = datasetLoss();
    std::vector<float> history = trainFor(net, 300, inputs, targets);
    double after = datasetLoss();

    ASSERT_FINITE(after);
    if (!(after < before))
        FAIL_WITH_NOTE("300 epochs on a separable dataset should reduce the mean loss",
                       "mean loss before %.12g, after %.12g", before, after);
    // The reported history must track the same descent.
    ASSERT_TRUE(history.back() < history.front());
}

TEST(Train_VerboseStillTrains)
{
    // verbose only adds printing; it must not change the numbers.
    silenceStdout();
    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}};
    std::vector<std::vector<float>> targets = {{0.0f}, {1.0f}};

    auto run = [&inputs, &targets](bool verbose) {
        core::Network net(1, 1, 0.5f, 1);
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5f, -0.3f});
        setRow(*net.hiddenLayers[0], 1, {0.2f, 0.8f});
        setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});
        std::vector<float> history(25);
        net.train(inputs, targets, 25, history.data(), 25, verbose, false, std::nullopt, 5);
        return history;
    };

    std::vector<float> quiet = run(false);
    std::vector<float> loud = run(true);
    for (size_t i = 0; i < quiet.size(); ++i)
        ASSERT_NEAR_AT(i, quiet[i], loud[i], kExact);
}

TEST(Train_MismatchedInputAndTargetCountsIsRejected)
{
    core::Network net(1, 1, 0.05f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}, {0.8f, 0.9f}};
    std::vector<std::vector<float>> targets = {{0.0f}};  // one short

    bool threw = false;
    try { trainFor(net, 5, inputs, targets); }
    catch (const std::invalid_argument&) { threw = true; }

    if (!threw)
        FAIL_WITH_NOTE(
            "train() must reject datasets whose input and target counts disagree instead of "
            "indexing past the end of the shorter one",
            "train() with %zu inputs and %zu targets returned normally",
            inputs.size(), targets.size());
}

TEST(Train_EmptyDatasetIsANoOp)
{
    core::Network net(1, 1, 0.05f);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);
    setRow(*net.outputLayers[0], 0, {0.7f, -0.4f});
    std::vector<float> before = net.outputLayers[0]->neuronWeights;

    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
    trainFor(net, 10, inputs, targets);

    for (size_t i = 0; i < before.size(); ++i)
        ASSERT_NEAR_AT(i, net.outputLayers[0]->neuronWeights[i], before[i], kExact);
}

TEST(Train_ZeroEpochsLeavesWeightsUntouched)
{
    Fixture f;
    std::vector<float> outBefore = f.net.outputLayers[0]->neuronWeights;
    std::vector<float> hidBefore = f.net.hiddenLayers[0]->neuronWeights;

    std::vector<std::vector<float>> inputs = {{0.1f, 0.2f}};
    std::vector<std::vector<float>> targets = {{1.0f}};

    trainFor(f.net, 0, inputs, targets);

    for (size_t i = 0; i < outBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.outputLayers[0]->neuronWeights[i], outBefore[i], kExact);
    for (size_t i = 0; i < hidBefore.size(); ++i)
        ASSERT_NEAR_AT(i, f.net.hiddenLayers[0]->neuronWeights[i], hidBefore[i], kExact);
}

// ---------------------------------------------------------------------------
// G. lifetime
// ---------------------------------------------------------------------------

TEST(Destructor_ReleasesLayersWithoutDoubleFree)
{
    // ASan is the real assertion here: a leak, a double delete or a delete of an
    // unfilled slot shows up in the --asan build.
    {
        core::Network net(2, 1, 0.01f);
        addSigmoidHidden(net, 0, 3, 2);
        addSigmoidHidden(net, 1, 2, 3);
        addSigmoidOutput(net, 0, 1, 2);
        net.initializeAllWeightsAndBiases();
    }
    ASSERT_TRUE(true);
}

TEST(InitializeAllWeightsAndBiases_SkipsUnfilledSlots)
{
    // A network constructed for N hidden layers but given fewer holds a null in
    // the remaining slots. The destructor handles that fine (delete nullptr is a
    // no-op), so this loop must skip nulls rather than dereference one.
    core::Network net(2, 1, 0.01f);
    addSigmoidHidden(net, 0, 3, 2);
    // slot 1 deliberately left null
    addSigmoidOutput(net, 0, 1, 3);

    net.initializeAllWeightsAndBiases();

    ASSERT_TRUE(net.hiddenLayers[1] == nullptr);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    return testing::runAll("core/network.h", argc > 1 ? argv[1] : nullptr);
}
