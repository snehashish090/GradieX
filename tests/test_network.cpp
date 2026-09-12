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

#include "test_framework.h"
#include "../core/network.h"

#include <vector>

namespace act = core::functions::activations;
namespace init = core::functions::initializers;
namespace loss = core::functions::loss;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static constexpr double kTol = 1e-9;

// Reference sigmoid / derivative, straight from the definition.
static double sig(double z) { return 1.0 / (1.0 + std::exp(-z)); }
static double sigPrime(double z) { double s = sig(z); return s * (1.0 - s); }

// train() and the layer print helpers write to stdout; tests that call them
// only care about the numbers.
static void silenceStdout() { std::freopen("/dev/null", "w", stdout); }

static void addSigmoidHidden(core::Network& net, int index, int width, int inputSize)
{
    net.addHiddenLayer(index, width, inputSize,
                       act::sigmoid, act::sigmoidDerivative, init::zeros,
                       loss::mse, loss::mseDerivative);
}

static void addSigmoidOutput(core::Network& net, int index, int width, int inputSize)
{
    net.addOutputLayer(index, width, inputSize,
                       act::sigmoid, act::sigmoidDerivative, init::zeros,
                       loss::mse, loss::mseDerivative);
}

static void setRow(core::Layer& layer, int neuron, std::vector<double> row)
{
    for (int i = 0; i < static_cast<int>(row.size()); ++i)
        layer.neuronWeights[neuron * layer.inputSize + i] = row[i];
}

// The fixture every forward/backward test below shares:
//
//   x = [0.1, 0.2]              target = [1.0]        learningRate = 0.5
//   hidden (2x2, sigmoid)  W = [[0.5,-0.3],[0.2,0.8]]  b = 0
//   output (1x2, sigmoid)  W = [[0.7,-0.4]]            b = 0
//
struct Fixture
{
    core::Network net{1, 1, 0.5};
    std::vector<double> input{0.1, 0.2};
    std::vector<double> target{1.0};

    // hand-derived reference values for that geometry
    double zh0, zh1, ah0, ah1, zo, ao;

    Fixture()
    {
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5, -0.3});
        setRow(*net.hiddenLayers[0], 1, {0.2, 0.8});
        setRow(*net.outputLayers[0], 0, {0.7, -0.4});

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
    core::Network net(3, 2, 0.05);
    ASSERT_EQ_INT(net.numHiddenLayers, 3);
    ASSERT_EQ_INT(net.numOutputLayers, 2);
    ASSERT_NEAR(net.learningRate, 0.05, kTol);
    ASSERT_EQ_INT(net.hiddenLayers.size(), 3);
    ASSERT_EQ_INT(net.outputLayers.size(), 2);
    // resize() value-initialises the pointers, so the destructor is safe on
    // slots that were never filled.
    for (auto* layer : net.hiddenLayers) ASSERT_TRUE(layer == nullptr);
    for (auto* layer : net.outputLayers) ASSERT_TRUE(layer == nullptr);
}

TEST(DefaultConstructor_LayerCountsMatchAllocation)
{
    // The default constructor sets numOutputLayers = 1 but resizes nothing, so
    // the counts disagree with the vectors. Anything that trusts the counts --
    // backwardPass() indexes outputLayers[0] unconditionally -- reads past the
    // end of an empty vector.
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
    core::Network net(2, 1, 0.01);
    addSigmoidHidden(net, 0, 4, 2);
    addSigmoidHidden(net, 1, 3, 4);

    ASSERT_TRUE(net.hiddenLayers[0] != nullptr);
    ASSERT_EQ_INT(net.hiddenLayers[0]->layerWidth, 4);
    ASSERT_EQ_INT(net.hiddenLayers[0]->inputSize, 2);
    ASSERT_EQ_INT(net.hiddenLayers[1]->layerWidth, 3);
    ASSERT_EQ_INT(net.hiddenLayers[1]->inputSize, 4);
    ASSERT_TRUE(net.hiddenLayers[0]->activationFunction == act::sigmoid);
}

TEST(AddOutputLayer_StoresLayerWithRequestedGeometry)
{
    core::Network net(0, 2, 0.01);
    addSigmoidOutput(net, 0, 1, 3);
    addSigmoidOutput(net, 1, 5, 3);

    ASSERT_EQ_INT(net.outputLayers[0]->layerWidth, 1);
    ASSERT_EQ_INT(net.outputLayers[0]->inputSize, 3);
    ASSERT_EQ_INT(net.outputLayers[1]->layerWidth, 5);
    ASSERT_TRUE(net.outputLayers[0]->lossFunction == loss::mse);
}

TEST(AddHiddenLayer_OutOfRangeIndexIsRejected)
{
    // hiddenLayers[layerIndex] is an unchecked operator[] write. Index 2 in a
    // 1-slot network scribbles past the end of the vector (ASan: heap overflow)
    // and leaks the layer it just allocated.
    core::Network net(1, 1, 0.01);
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
            "addHiddenLayer/addOutputLayer should bounds-check layerIndex (throw, or use .at()) "
            "instead of writing past the end of the layer vector",
            "addHiddenLayer(index=2) on a network with %zu hidden slots returned normally",
            net.hiddenLayers.size());
}

TEST(InitializeAllWeightsAndBiases_ReachesEveryLayer)
{
    core::Network net(2, 1, 0.01);
    net.addHiddenLayer(0, 4, 2, act::sigmoid, act::sigmoidDerivative, init::xavier,
                       loss::mse, loss::mseDerivative);
    net.addHiddenLayer(1, 3, 4, act::sigmoid, act::sigmoidDerivative, init::xavier,
                       loss::mse, loss::mseDerivative);
    net.addOutputLayer(0, 1, 3, act::sigmoid, act::sigmoidDerivative, init::xavier,
                       loss::mse, loss::mseDerivative);

    const double sentinel = 1e300;
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
    double first = f.net.outputLayers[0]->neuronOutputsActivated[0];

    std::vector<double> other{0.9, 0.4};
    f.net.forwardPass(other.data());
    f.net.forwardPass(f.input.data());
    double again = f.net.outputLayers[0]->neuronOutputsActivated[0];

    ASSERT_NEAR(again, first, kTol);
}

// ---------------------------------------------------------------------------
// C. backwardPass
// ---------------------------------------------------------------------------

TEST(BackwardPass_SetsOutputDeltaFromTarget)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    f.net.backwardPass(f.target);

    ASSERT_NEAR(f.net.outputLayers[0]->neuronDeltas[0], f.outputDelta(), kTol);
}

TEST(BackwardPass_PropagatesDeltaIntoHiddenLayer)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    f.net.backwardPass(f.target);

    core::HiddenLayer& hidden = *f.net.hiddenLayers[0];
    ASSERT_NEAR_AT(0, hidden.neuronDeltas[0], f.hiddenDelta(0), kTol);
    ASSERT_NEAR_AT(1, hidden.neuronDeltas[1], f.hiddenDelta(1), kTol);
}

TEST(BackwardPass_AppliesOneGradientStepToEveryWeight)
{
    Fixture f;
    const double lr = 0.5;
    const double outWBefore[2] = {0.7, -0.4};
    const double hiddenWBefore[2][2] = {{0.5, -0.3}, {0.2, 0.8}};

    f.net.forwardPass(f.input.data());
    f.net.backwardPass(f.target);

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

TEST(BackwardPass_RepeatedStepsConvergeOnOneSample)
{
    // 200 steps on a single (input, target) pair must drive the loss down; this
    // is the end-to-end check that every sign in the chain rule is right.
    Fixture f;
    f.net.forwardPass(f.input.data());
    double lossBefore = f.net.computeLoss(f.target);

    for (int step = 0; step < 200; ++step)
    {
        f.net.forwardPass(f.input.data());
        f.net.backwardPass(f.target);
    }
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
    std::vector<double> selfTarget{f.net.outputLayers[0]->neuronOutputsActivated[0]};

    ASSERT_NEAR(f.net.computeLoss(selfTarget), 0.0, kTol);
}

TEST(ComputeLoss_IsPositiveAndGrowsWithError)
{
    Fixture f;
    f.net.forwardPass(f.input.data());
    std::vector<double> near{f.ao + 0.1};
    std::vector<double> far{f.ao + 0.4};

    double lossNear = f.net.computeLoss(near);
    double lossFar = f.net.computeLoss(far);
    ASSERT_TRUE(lossNear > 0.0);
    ASSERT_TRUE(lossFar > lossNear);
}

TEST(ComputeLoss_AgreesWithTheConfiguredLossFunction)
{
    // computeLoss() hardcodes a sum of squared errors. The layer carries a
    // lossFunction pointer, and backwardPass() differentiates *that*, so the
    // number reported during training should come from the same function.
    // With mse and a 3-wide output the two differ by the 1/N factor.
    core::Network net(0, 1, 0.01);
    addSigmoidOutput(net, 0, 3, 2);
    net.outputLayers[0]->neuronOutputsActivated = {0.5, 0.25, 0.75};

    std::vector<double> target{1.0, 0.0, 0.0};
    double reported = net.computeLoss(target);
    double configured = loss::mse(target.data(),
                                  net.outputLayers[0]->neuronOutputsActivated.data(), 3);

    if (!testing::nearlyEqual(reported, configured, 1e-12))
        FAIL_WITH_NOTE(
            "computeLoss() should call the layer's lossFunction rather than hardcoding "
            "a squared-error sum, otherwise the reported loss is not the quantity being minimised",
            "computeLoss() = %.12g, layer's mse() = %.12g", reported, configured);
}

// ---------------------------------------------------------------------------
// E. train
// ---------------------------------------------------------------------------

TEST(Train_MoreEpochsThanSamplesStaysInBounds)
{
    // Regression test: train() used to index the dataset by the epoch counter,
    // so it read past the end of the sample vector on epoch == numSamples.
    // 6 samples, 100 epochs.
    silenceStdout();
    core::Network net(1, 1, 0.05);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);
    net.initializeAllWeightsAndBiases();

    std::vector<std::vector<double>> inputs = {
        {0.1, 0.2}, {0.8, 0.9}, {0.2, 0.1}, {0.9, 0.7}, {0.4, 0.5}, {0.6, 0.4}};
    std::vector<std::vector<double>> targets = {
        {0.0}, {1.0}, {0.0}, {1.0}, {0.0}, {1.0}};

    net.train(inputs, targets, 100);

    for (double w : net.outputLayers[0]->neuronWeights) ASSERT_FINITE(w);
    for (double w : net.hiddenLayers[0]->neuronWeights) ASSERT_FINITE(w);
}

TEST(Train_OneEpochVisitsEverySampleInOrder)
{
    // Two identically-weighted networks: one trained for a single epoch, the
    // other stepped by hand over each sample. They must end up identical.
    silenceStdout();
    std::vector<std::vector<double>> inputs = {{0.1, 0.2}, {0.8, 0.9}, {0.4, 0.5}};
    std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {1.0}};

    auto build = [](core::Network& net) {
        addSigmoidHidden(net, 0, 2, 2);
        addSigmoidOutput(net, 0, 1, 2);
        setRow(*net.hiddenLayers[0], 0, {0.5, -0.3});
        setRow(*net.hiddenLayers[0], 1, {0.2, 0.8});
        setRow(*net.outputLayers[0], 0, {0.7, -0.4});
    };

    core::Network trained(1, 1, 0.5);
    core::Network stepped(1, 1, 0.5);
    build(trained);
    build(stepped);

    trained.train(inputs, targets, 1);
    for (size_t sample = 0; sample < inputs.size(); ++sample)
        stepped.runEpoch(inputs[sample], targets[sample]);

    for (size_t i = 0; i < stepped.hiddenLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, trained.hiddenLayers[0]->neuronWeights[i],
                       stepped.hiddenLayers[0]->neuronWeights[i], kTol);
    for (size_t i = 0; i < stepped.outputLayers[0]->neuronWeights.size(); ++i)
        ASSERT_NEAR_AT(i, trained.outputLayers[0]->neuronWeights[i],
                       stepped.outputLayers[0]->neuronWeights[i], kTol);
}

TEST(Train_LowersLossAcrossEpochs)
{
    silenceStdout();
    core::Network net(1, 1, 0.5);
    addSigmoidHidden(net, 0, 3, 2);
    addSigmoidOutput(net, 0, 1, 3);
    setRow(*net.hiddenLayers[0], 0, {0.5, -0.3});
    setRow(*net.hiddenLayers[0], 1, {0.2, 0.8});
    setRow(*net.hiddenLayers[0], 2, {-0.6, 0.4});
    setRow(*net.outputLayers[0], 0, {0.7, -0.4, 0.2});

    std::vector<std::vector<double>> inputs = {{0.1, 0.2}, {0.8, 0.9}, {0.2, 0.1}, {0.9, 0.7}};
    std::vector<std::vector<double>> targets = {{0.0}, {1.0}, {0.0}, {1.0}};

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
    net.train(inputs, targets, 300);
    double after = datasetLoss();

    ASSERT_FINITE(after);
    if (!(after < before))
        FAIL_WITH_NOTE("300 epochs on a separable dataset should reduce the mean loss",
                       "mean loss before %.12g, after %.12g", before, after);
}

TEST(Train_MismatchedInputAndTargetCountsIsRejected)
{
    silenceStdout();
    core::Network net(1, 1, 0.05);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);

    std::vector<std::vector<double>> inputs = {{0.1, 0.2}, {0.8, 0.9}};
    std::vector<std::vector<double>> targets = {{0.0}};  // one short

    bool threw = false;
    try { net.train(inputs, targets, 5); }
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
    silenceStdout();
    core::Network net(1, 1, 0.05);
    addSigmoidHidden(net, 0, 2, 2);
    addSigmoidOutput(net, 0, 1, 2);
    setRow(*net.outputLayers[0], 0, {0.7, -0.4});

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    net.train(inputs, targets, 10);

    ASSERT_NEAR_AT(0, net.outputLayers[0]->neuronWeights[0], 0.7, kTol);
    ASSERT_NEAR_AT(1, net.outputLayers[0]->neuronWeights[1], -0.4, kTol);
}

TEST(Train_ZeroEpochsLeavesWeightsUntouched)
{
    silenceStdout();
    Fixture f;
    std::vector<std::vector<double>> inputs = {{0.1, 0.2}};
    std::vector<std::vector<double>> targets = {{1.0}};

    f.net.train(inputs, targets, 0);

    ASSERT_NEAR_AT(0, f.net.outputLayers[0]->neuronWeights[0], 0.7, kTol);
    ASSERT_NEAR_AT(1, f.net.outputLayers[0]->neuronWeights[1], -0.4, kTol);
    ASSERT_NEAR_AT(0, f.net.hiddenLayers[0]->neuronWeights[0], 0.5, kTol);
}

// ---------------------------------------------------------------------------
// F. lifetime
// ---------------------------------------------------------------------------

TEST(Destructor_ReleasesLayersWithoutDoubleFree)
{
    // ASan is the real assertion here: a leak, a double delete or a delete of an
    // unfilled slot shows up in the --asan build.
    {
        core::Network net(2, 1, 0.01);
        addSigmoidHidden(net, 0, 3, 2);
        addSigmoidHidden(net, 1, 2, 3);
        addSigmoidOutput(net, 0, 1, 2);
        net.initializeAllWeightsAndBiases();
    }
    ASSERT_TRUE(true);
}

TEST(InitializeAllWeightsAndBiases_SkipsUnfilledSlots)
{
    // A network constructed for N hidden layers but given fewer dereferences a
    // null pointer here. The destructor handles the same case fine (delete
    // nullptr is a no-op), so this loop should skip nulls too.
    core::Network net(2, 1, 0.01);
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
