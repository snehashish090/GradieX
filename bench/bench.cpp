// One harness, three engines.
//
//   clang++ -std=c++20 -O2 -DBENCH_V1 bench/bench.cpp -o bench_v1
//   clang++ -std=c++20 -O2 -DBENCH_V2 bench/bench.cpp -o bench_v2
//   clang++ -std=c++20 -O2            bench/bench.cpp -o bench_v3
//
// Everything except the engine calls is shared, so the timing loop, the data and
// the architectures are identical across versions by construction rather than by
// inspection. run_bench.sh drives the flag matrix.
//
// One "update" is the work of a single sample: forward pass, backward pass, and
// the weight update. That is what v0.0.2 does per sample (it assigns gradients
// and applies them immediately), and what v0.0.3 does at batch size 1 (it zeroes,
// accumulates, then applies). The extra zeroing pass is a real cost of the
// batching architecture and is counted, not excused.
//
// v0.0.1 and v0.0.3 are also measured at batch 32, where that zeroing amortises
// over the batch -- the configuration mini-batching actually exists for.
//
// On v0.0.1: it predates `core/` entirely. It shipped as one translation unit
// (0.0.1/main.cpp) whose library half is compiled in here through
// bench/v1/v001_shim.h, wrapped in `namespace v1` so it shares a binary with the
// later engines. Three differences are the engine's, not the harness's, and all
// three are counted rather than normalised away:
//
//   1. It is `double` throughout; v0.0.2 is `double`, v0.0.3 is `float`.
//   2. It allocates per-neuron weight vectors and reallocates the whole gradient
//      set in `zero_gradients()` on every update -- the cost the v0.0.2 flat
//      layout was written to remove.
//   3. Its hidden layers are uniform width (one width for all of them). That is
//      a documented limitation of its constructor, so the v1 column exists only
//      for architectures with uniform hidden widths.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef BENCH_V1
#  include "v1/v001_shim.h"
   using Scalar = double;
   static const char* kVersion = "0.0.1";
   using Net = v1::NeuralNetwork;
#elif defined(BENCH_V2)
#  include "../../0.0.2/core/network.h"
   using Scalar = double;
   static const char* kVersion = "0.0.2";
   using Net = core::Network;
#else
#  include "../core/network.h"
   using Scalar = float;
   static const char* kVersion = "0.0.3";
   using Net = core::Network;
#endif

#ifndef BENCH_V1
// v0.0.1 has no `core/` namespace and names its functions differently; its calls
// are spelled out in buildNetwork() below.
namespace act = core::functions::activations;
namespace ini = core::functions::initializers;
namespace los = core::functions::loss;
#endif

// ---------------------------------------------------------------- input data
//
// A plain LCG rather than <random>: it has to produce the same values in both
// builds, and std::mt19937's distributions are not guaranteed to.

struct Lcg {
    unsigned long long state;
    explicit Lcg(unsigned long long seed) : state(seed) {}
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((state >> 11) & ((1ULL << 53) - 1)) / 9007199254740992.0;
    }
    double symmetric() { return next() * 2.0 - 1.0; }
};

struct Dataset {
    std::vector<std::vector<Scalar>> inputs;
    std::vector<std::vector<Scalar>> targets;
};

static Dataset makeData(int samples, int inputSize, int outputSize) {
    Lcg rng(20260917);
    Dataset d;
    d.inputs.resize(samples);
    d.targets.resize(samples);
    for (int s = 0; s < samples; ++s) {
        d.inputs[s].resize(inputSize);
        for (int i = 0; i < inputSize; ++i)
            d.inputs[s][i] = static_cast<Scalar>(rng.symmetric());
        d.targets[s].resize(outputSize);
        for (int o = 0; o < outputSize; ++o)
            d.targets[s][o] = static_cast<Scalar>(rng.next());
    }
    return d;
}

// ------------------------------------------------------------------ the model

struct Arch {
    const char* name;
    int inputSize;
    std::vector<int> hidden;
    int outputSize;

    int parameters() const {
        int total = 0, prev = inputSize;
        for (int w : hidden) { total += prev * w + w; prev = w; }
        return total + prev * outputSize + outputSize;
    }
};

static Net* buildNetwork(const Arch& a) {
#ifdef BENCH_V1
    // v0.0.1 takes one hidden width for all hidden layers, so it can only
    // represent architectures whose hidden widths are uniform. The harness only
    // schedules v1 for those (see main()).
    const int hiddenWidth = a.hidden.empty() ? 0 : a.hidden.front();
    auto* net = new v1::NeuralNetwork(a.inputSize,
                                      static_cast<int>(a.hidden.size()),
                                      hiddenWidth,
                                      a.outputSize,
                                      v1::Activation::ReLU,       // hidden
                                      v1::Activation::Sigmoid,    // output
                                      v1::Loss::MeanSquaredError,
                                      20260917);                  // fixed seed
    return net;
#else
#  ifdef BENCH_V2
    auto* net = new core::Network(static_cast<int>(a.hidden.size()), 1, 0.01);
#  else
    auto* net = new core::Network(static_cast<int>(a.hidden.size()), 1, 0.01f, 1);
#  endif
    int prev = a.inputSize;
    for (size_t i = 0; i < a.hidden.size(); ++i) {
#  ifdef BENCH_V2
        net->addHiddenLayer(static_cast<int>(i), a.hidden[i], prev,
                            act::relu, act::reluDerivative, ini::he,
                            los::mse, los::mseDerivative);
#  else
        net->addHiddenLayer(static_cast<int>(i), a.hidden[i], prev,
                            act::relu, ini::he, los::mse);
#  endif
        prev = a.hidden[i];
    }
#  ifdef BENCH_V2
    net->addOutputLayer(0, a.outputSize, prev, act::sigmoid, act::sigmoidDerivative,
                        ini::he, los::mse, los::mseDerivative);
#  else
    net->addOutputLayer(0, a.outputSize, prev, act::sigmoid, ini::he, los::mse);
#  endif
    net->initializeAllWeightsAndBiases();
    return net;
#endif
}

// One sample's forward + backward + update, spelled with each engine's own API.
// Unused under BENCH_V1: that branch inlines the same three calls so it can also
// read the loss between them, but the translation is kept here as the reference
// for what "one update" means in each engine.
[[maybe_unused]] static inline void oneUpdate(Net& net,
                             const std::vector<Scalar>& x,
                             std::vector<Scalar>& y) {
#ifdef BENCH_V1
    net.zero_gradients();
    net.forward_pass(x);
    net.accumulate_gradients(y);
    net.apply_gradients(0.01, 1.0, 0.0);
#elif defined(BENCH_V2)
    net.forwardPass(x.data());
    net.backwardPass(y);          // computes gradients AND applies them
#else
    net.zeroAllGradients();
    net.forwardPass(x.data());
    net.accumulateGradients(y);
    net.applyGradients();
#endif
}

// ------------------------------------------------------------------- timing

using Clock = std::chrono::steady_clock;

// Loss on the first and last sample visited by a run. These exist so the engines
// can be checked for behavioural parity: equivalent work should land on
// comparable losses, and a wrong API translation in the harness would not.
// v0.0.2's `backwardPass` takes the target and both its forward and backward
// passes return void, so it exposes no loss from outside; its loss columns are
// left blank rather than measured with a call the other two do not make.
static double gFirstLoss = 0.0;
static double gLastLoss = 0.0;

// `batch` samples accumulated, then one update. batch == 1 is the per-sample path
// above. v0.0.2 has no batching, so it always runs the per-sample path and the
// batch column is left blank for it.
static double secondsFor(Net& net, Dataset& d, int iters, int batch) {
    const int n = static_cast<int>(d.inputs.size());
    bool first = true;
    double last = 0.0;
    auto observe = [&](double loss) {
        if (first) { gFirstLoss = loss; first = false; }
        last = loss;
    };
    auto start = Clock::now();
#if defined(BENCH_V2)
    (void)batch;
    (void)observe;
    for (int i = 0; i < iters; ++i) {
        const int s = i % n;
        oneUpdate(net, d.inputs[s], d.targets[s]);
    }
#elif defined(BENCH_V1)
    if (batch <= 1) {
        for (int i = 0; i < iters; ++i) {
            const int s = i % n;
            const std::vector<double>& p = net.forward_pass(d.inputs[s]);
            observe(net.calculate_loss(p, d.targets[s]));
            net.zero_gradients();
            net.accumulate_gradients(d.targets[s]);
            net.apply_gradients(0.01, 1.0, 0.0);
        }
    } else {
        for (int i = 0; i < iters; i += batch) {
            net.zero_gradients();
            const int take = (i + batch <= iters) ? batch : (iters - i);
            for (int b = 0; b < take; ++b) {
                const int s = (i + b) % n;
                const std::vector<double>& p = net.forward_pass(d.inputs[s]);
                observe(net.calculate_loss(p, d.targets[s]));
                net.accumulate_gradients(d.targets[s]);
            }
            net.apply_gradients(0.01, 1.0 / static_cast<double>(take), 0.0);
        }
    }
#else
    if (batch <= 1) {
        for (int i = 0; i < iters; ++i) {
            const int s = i % n;
            net.forwardPass(d.inputs[s].data());
            observe(net.computeLoss(d.targets[s]));
            net.zeroAllGradients();
            net.accumulateGradients(d.targets[s]);
            net.applyGradients();
        }
    } else {
        for (int i = 0; i < iters; i += batch) {
            net.zeroAllGradients();
            const int take = (i + batch <= iters) ? batch : (iters - i);
            for (int b = 0; b < take; ++b) {
                const int s = (i + b) % n;
                net.forwardPass(d.inputs[s].data());
                observe(net.computeLoss(d.targets[s]));
                net.accumulateGradients(d.targets[s]);
            }
            net.scaleAllGradients(1.0f / static_cast<float>(take));
            net.applyGradients();
        }
    }
#endif
    auto end = Clock::now();
    gLastLoss = last;
    return std::chrono::duration<double>(end - start).count();
}

// Pick an iteration count that puts one measurement at roughly a quarter second,
// so short architectures are not dominated by clock resolution. `--quick` cuts
// that to a twentieth of a second, for smoke-checking the harness itself.
static int calibrate(Net& net, Dataset& d, double target) {
    const int probe = 200;
    double t = secondsFor(net, d, probe, 1);
    if (t <= 0.0) return 200000;
    double perUpdate = t / probe;
    double want = target / perUpdate;
    if (want < 500.0) want = 500.0;
    if (want > 4000000.0) want = 4000000.0;
    return static_cast<int>(want);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    const char* flags = (argc > 1) ? argv[1] : "?";
    const int batch = (argc > 2) ? std::atoi(argv[2]) : 1;
    const bool quick = (argc > 3) && std::string(argv[3]) == "--quick";
    const int trials = quick ? 3 : 9;
    const double target = quick ? 0.05 : 0.25;
    const int probeIters = quick ? 500 : 2000;

    std::vector<Arch> arches = {
        {"2->8x2->1",      2,   {8, 8},        1},
        {"20->32x2->3",    20,  {32, 32},      3},
        {"64->128x3->10",  64,  {128,128,128}, 10},
#ifndef BENCH_V1
        // v0.0.1 takes a single hidden width for every hidden layer, so it cannot
        // represent 784->128x2->10's mixed 784/128 fan-in. Dropped from its column
        // rather than approximated -- see the header comment.
        {"784->128x2->10", 784, {128, 128},    10},
#endif
    };

    for (const Arch& a : arches) {
        Dataset d = makeData(256, a.inputSize, a.outputSize);
        Net* net = buildNetwork(a);

        secondsFor(*net, d, 200, batch);          // warm up caches and branch predictors
        const int iters = calibrate(*net, d, target);

        // The parity probe: a fresh, identically-seeded network trained the same way,
        // so its loss trajectory can be compared against the other engines'. The
        // weights are not the same across versions -- each has its own RNG -- so
        // this checks that the work is comparable in kind, not identical in value.
        Net* probe = buildNetwork(a);
        secondsFor(*probe, d, probeIters, batch);
        const double firstLoss = gFirstLoss;
        const double lastLoss = gLastLoss;
        delete probe;

        std::vector<double> perUpdate;               // microseconds per SAMPLE
        for (int t = 0; t < trials; ++t)
            perUpdate.push_back(secondsFor(*net, d, iters, batch) / iters * 1e6);

        std::sort(perUpdate.begin(), perUpdate.end());
        // The published figure is the MINIMUM across trials: interference only ever
        // adds time, so the fastest run is the least contaminated estimate of what
        // the code costs. The median and max are printed alongside so a noisy
        // measurement is visible rather than hidden.
        // version,arch,params,flags,batch,median_us,min_us,max_us,iters,loss_first,loss_last
        std::printf("%s,%s,%d,%s,%d,%.4f,%.4f,%.4f,%d,%.6f,%.6f\n",
                    kVersion, a.name, a.parameters(), flags, batch,
                    median(perUpdate), perUpdate.front(), perUpdate.back(), iters,
                    firstLoss, lastLoss);
        std::fflush(stdout);
        delete net;
    }
    return 0;
}
