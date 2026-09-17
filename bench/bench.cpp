// One harness, two engines.
//
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
// v0.0.3 is also measured at batch 32, where that zeroing amortises over the
// batch -- the configuration mini-batching actually exists for.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef BENCH_V2
#  include "../../0.0.2/core/network.h"
   using Scalar = double;
   static const char* kVersion = "0.0.2";
#else
#  include "../core/network.h"
   using Scalar = float;
   static const char* kVersion = "0.0.3";
#endif

namespace act = core::functions::activations;
namespace ini = core::functions::initializers;
namespace los = core::functions::loss;

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

static core::Network* buildNetwork(const Arch& a) {
#ifdef BENCH_V2
    auto* net = new core::Network(static_cast<int>(a.hidden.size()), 1, 0.01);
#else
    auto* net = new core::Network(static_cast<int>(a.hidden.size()), 1, 0.01f, 1);
#endif
    int prev = a.inputSize;
    for (size_t i = 0; i < a.hidden.size(); ++i) {
#ifdef BENCH_V2
        net->addHiddenLayer(static_cast<int>(i), a.hidden[i], prev,
                            act::relu, act::reluDerivative, ini::he,
                            los::mse, los::mseDerivative);
#else
        net->addHiddenLayer(static_cast<int>(i), a.hidden[i], prev,
                            act::relu, ini::he, los::mse);
#endif
        prev = a.hidden[i];
    }
#ifdef BENCH_V2
    net->addOutputLayer(0, a.outputSize, prev, act::sigmoid, act::sigmoidDerivative,
                        ini::he, los::mse, los::mseDerivative);
#else
    net->addOutputLayer(0, a.outputSize, prev, act::sigmoid, ini::he, los::mse);
#endif
    net->initializeAllWeightsAndBiases();
    return net;
}

// One sample's forward + backward + update, spelled with each engine's own API.
static inline void oneUpdate(core::Network& net,
                             const std::vector<Scalar>& x,
                             std::vector<Scalar>& y) {
#ifdef BENCH_V2
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

// `batch` samples accumulated, then one update. batch == 1 is the per-sample path
// above. v0.0.2 has no batching, so it always runs the per-sample path and the
// batch column is left blank for it.
static double secondsFor(core::Network& net, Dataset& d, int iters, int batch) {
    const int n = static_cast<int>(d.inputs.size());
    auto start = Clock::now();
#ifdef BENCH_V2
    (void)batch;
    for (int i = 0; i < iters; ++i) {
        const int s = i % n;
        oneUpdate(net, d.inputs[s], d.targets[s]);
    }
#else
    if (batch <= 1) {
        for (int i = 0; i < iters; ++i) {
            const int s = i % n;
            oneUpdate(net, d.inputs[s], d.targets[s]);
        }
    } else {
        for (int i = 0; i < iters; i += batch) {
            net.zeroAllGradients();
            const int take = (i + batch <= iters) ? batch : (iters - i);
            for (int b = 0; b < take; ++b) {
                const int s = (i + b) % n;
                net.forwardPass(d.inputs[s].data());
                net.accumulateGradients(d.targets[s]);
            }
            net.scaleAllGradients(1.0f / static_cast<float>(take));
            net.applyGradients();
        }
    }
#endif
    auto end = Clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// Pick an iteration count that puts one measurement at roughly a quarter second,
// so short architectures are not dominated by clock resolution.
static int calibrate(core::Network& net, Dataset& d) {
    const int probe = 200;
    double t = secondsFor(net, d, probe, 1);
    if (t <= 0.0) return 200000;
    double perUpdate = t / probe;
    double want = 0.25 / perUpdate;
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
    const int trials = 9;

    std::vector<Arch> arches = {
        {"2->8x2->1",      2,   {8, 8},        1},
        {"20->32x2->3",    20,  {32, 32},      3},
        {"64->128x3->10",  64,  {128,128,128}, 10},
        {"784->128x2->10", 784, {128, 128},    10},
    };

    for (const Arch& a : arches) {
        Dataset d = makeData(256, a.inputSize, a.outputSize);
        core::Network* net = buildNetwork(a);

        secondsFor(*net, d, 200, batch);          // warm up caches and branch predictors
        const int iters = calibrate(*net, d);

        std::vector<double> perUpdate;               // microseconds per SAMPLE
        for (int t = 0; t < trials; ++t)
            perUpdate.push_back(secondsFor(*net, d, iters, batch) / iters * 1e6);

        std::sort(perUpdate.begin(), perUpdate.end());
        // The published figure is the MINIMUM across trials: interference only ever
        // adds time, so the fastest run is the least contaminated estimate of what
        // the code costs. The median and max are printed alongside so a noisy
        // measurement is visible rather than hidden.
        // version,arch,params,flags,batch,median_us,min_us,max_us,iters
        std::printf("%s,%s,%d,%s,%d,%.4f,%.4f,%.4f,%d\n",
                    kVersion, a.name, a.parameters(), flags, batch,
                    median(perUpdate), perUpdate.front(), perUpdate.back(), iters);
        std::fflush(stdout);
        delete net;
    }
    return 0;
}
