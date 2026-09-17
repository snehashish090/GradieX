// Python bindings for the GradieX 0.0.3 engine.
//
// Two design rules hold this file together.
//
// 1. Training happens in the engine. `train()` hands the whole dataset to
//    core::Network::train and gets a loss history back; there is no per-sample
//    Python-side or binding-side loop. Batching, gradient accumulation, the
//    shuffle and the update rule are all the engine's, so a model trained from
//    Python takes exactly the same code path as one trained from main.cpp.
//
// 2. The C++ API lets a caller reach states that crash or silently train wrong:
//    an out-of-range layer index, unfilled layer slots, an activation whose
//    derivative the engine cannot resolve, more than one output layer (the
//    backward pass only propagates from outputLayers[0]). This layer collects
//    layer specs first and constructs core::Network only in build(), so none of
//    those states are reachable from Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/network.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace act = core::functions::activations;
namespace ini = core::functions::initializers;
namespace los = core::functions::loss;

// ---------------------------------------------------------------- name tables
//
// Only the forward functions are named here. The engine derives the matching
// derivative itself (Layer's constructor calls fetchAssociatedDerivative), so
// naming a derivative here would just be a second source of truth to drift.

static const std::map<std::string, act::ActivationFunction>& activations() {
    static const std::map<std::string, act::ActivationFunction> t{
        {"identity",   act::identity},
        {"linear",     act::identity},
        {"sigmoid",    act::sigmoid},
        {"relu",       act::relu},
        {"leaky_relu", act::leakyRelu},
        {"tanh",       act::tanh},
        {"swish",      act::swish},
        {"gelu",       act::gelu},
        {"softmax",    act::softmax},
    };
    return t;
}
static const std::map<std::string, los::LossFunction>& losses() {
    static const std::map<std::string, los::LossFunction> t{
        {"mse",                   los::mse},
        {"mae",                   los::mae},
        {"huber",                 los::huber},
        {"binary_cross_entropy",  los::binaryCrossEntropy},
        {"bce",                   los::binaryCrossEntropy},
        {"cross_entropy",         los::crossEntropy},
        {"cce",                   los::crossEntropy},
        {"softmax_cross_entropy", los::softmaxCrossEntropy},
    };
    return t;
}
static const std::map<std::string, ini::initFunction>& initializers() {
    static const std::map<std::string, ini::initFunction> t{
        {"zeros", ini::zeros}, {"xavier", ini::xavier}, {"he", ini::he},
    };
    return t;
}

template <typename Map>
static std::string known_keys(const Map& m) {
    std::ostringstream os;
    bool first = true;
    for (const auto& kv : m) { if (!first) os << ", "; os << "'" << kv.first << "'"; first = false; }
    return os.str();
}

static act::ActivationFunction lookup_activation(const std::string& name, const char* where) {
    auto it = activations().find(name);
    if (it == activations().end())
        throw std::invalid_argument("unknown activation '" + name + "' for " + where +
                                    "; available: " + known_keys(activations()));
    // The engine resolves the derivative by function-pointer identity. If it cannot,
    // the layer would train against a null derivative and segfault on the first
    // backward pass, so refuse here instead.
    if (act::fetchAssociatedDerivative(it->second) == nullptr) {
        if (name == "softmax")
            throw std::invalid_argument(
                "activation 'softmax' cannot be trained through: its derivative is a Jacobian, "
                "and the engine implements only the fused form. Use activation='identity' with "
                "loss='softmax_cross_entropy', which computes dL/dz = softmax(z) - y directly. "
                "Note that predict() then returns logits, not probabilities.");
        throw std::invalid_argument("activation '" + name + "' has no derivative registered in "
                                    "core/activations.h, so it cannot be trained through");
    }
    return it->second;
}
static los::LossFunction lookup_loss(const std::string& name) {
    auto it = losses().find(name);
    if (it == losses().end())
        throw std::invalid_argument("unknown loss '" + name + "'; available: " + known_keys(losses()));
    if (los::fetchAssociatedDerivative(it->second) == nullptr)
        throw std::invalid_argument("loss '" + name + "' has no derivative registered in "
                                    "core/loss.h, so it cannot be trained against");
    return it->second;
}
static ini::initFunction lookup_init(const std::string& name) {
    auto it = initializers().find(name);
    if (it == initializers().end())
        throw std::invalid_argument("unknown initializer '" + name + "'; available: " +
                                    known_keys(initializers()));
    return it->second;
}

// ------------------------------------------------------------- numpy helpers
//
// The engine is float32 end to end, so the arrays crossing this boundary are
// float32. NumPy float64 input is accepted and cast (forcecast), because
// refusing a plain Python list of doubles would be hostile.

using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

static std::vector<float> to_vec(const py::object& o, const char* name) {
    Arr a = Arr::ensure(o);
    if (!a) throw std::invalid_argument(std::string(name) + " must be a sequence of numbers");
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string(name) + " must be 1-D, got " +
                                    std::to_string(a.ndim()) + "-D");
    const float* p = a.data();
    return std::vector<float>(p, p + a.size());
}

static std::vector<std::vector<float>> to_mat(const py::object& o, const char* name) {
    Arr a = Arr::ensure(o);
    if (!a) throw std::invalid_argument(std::string(name) + " must be a 2-D array of numbers");
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(name) + " must be 2-D (n_samples, n_features), got " +
                                    std::to_string(a.ndim()) + "-D");
    auto r = a.unchecked<2>();
    std::vector<std::vector<float>> out(static_cast<size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<size_t>(i)].resize(static_cast<size_t>(r.shape(1)));
        for (py::ssize_t j = 0; j < r.shape(1); ++j)
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = r(i, j);
    }
    return out;
}

static Arr to_numpy(const std::vector<float>& v) {
    Arr out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// ------------------------------------------------------------------- Network

class Network {
public:
    Network(double learning_rate, int batch_size) {
        if (!(learning_rate > 0.0))
            throw std::invalid_argument("learning_rate must be > 0");
        if (batch_size < 0)
            throw std::invalid_argument("batch_size must be >= 0 (0 means one batch per epoch)");
        lr_ = static_cast<float>(learning_rate);
        batch_ = batch_size;
    }
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    void add_hidden(int width, std::optional<int> input_size,
                    const std::string& activation, const std::string& init) {
        if (built_) throw std::runtime_error("cannot add layers after build()");
        if (has_output())
            throw std::runtime_error("hidden layers must be added before the output layer");
        Spec s;
        s.width = require_positive(width, "width");
        s.in = resolve_input_size(input_size);
        s.a = lookup_activation(activation, "a hidden layer");
        s.init = lookup_init(init);
        // Hidden layers never use their loss function; the engine stores one anyway.
        s.l = los::mse;
        s.is_output = false;
        s.act_name = activation;
        s.init_name = init;
        specs_.push_back(s);
    }

    void add_output(int width, std::optional<int> input_size,
                    const std::string& activation, const std::string& loss,
                    const std::string& init) {
        if (built_) throw std::runtime_error("cannot add layers after build()");
        if (has_output())
            throw std::runtime_error(
                "only one output layer is supported: the engine's accumulateGradients "
                "propagates from outputLayers[0] only, so additional output layers would "
                "train silently incorrectly");
        Spec s;
        s.width = require_positive(width, "width");
        s.in = resolve_input_size(input_size);
        s.a = lookup_activation(activation, "the output layer");
        s.l = lookup_loss(loss);
        s.init = lookup_init(init);
        s.is_output = true;
        s.act_name = activation;
        s.loss_name = loss;
        s.init_name = init;
        specs_.push_back(s);
    }

    void build() {
        if (built_) throw std::runtime_error("build() has already been called");
        if (!has_output())
            throw std::runtime_error("add an output layer with add_output() before build()");
        int n_hidden = 0;
        for (const auto& s : specs_) if (!s.is_output) ++n_hidden;

        net_ = std::make_unique<core::Network>(n_hidden, 1, lr_, batch_);
        int idx = 0;
        for (const auto& s : specs_) {
            if (s.is_output) net_->addOutputLayer(0,      s.width, s.in, s.a, s.init, s.l);
            else             net_->addHiddenLayer(idx++,  s.width, s.in, s.a, s.init, s.l);
        }
        net_->initializeAllWeightsAndBiases();
        built_ = true;
    }

    // -------- single-sample primitives, one per engine call

    Arr forward(const py::object& x) {
        require_built();
        std::vector<float> in = to_vec(x, "x");
        check_len(in.size(), static_cast<size_t>(specs_.front().in), "x", "the first layer's input_size");
        net_->forwardPass(in.data());
        return to_numpy(output_layer()->neuronOutputsActivated);
    }

    void zero_gradients() { require_built(); net_->zeroAllGradients(); }

    void accumulate(const py::object& y) {
        require_built();
        std::vector<float> t = check_target(y);
        net_->accumulateGradients(t);
    }

    void apply_gradients() { require_built(); net_->applyGradients(); }

    void scale_gradients(double factor) {
        require_built();
        net_->scaleAllGradients(static_cast<float>(factor));
    }

    // Convenience: the whole update for the sample just forwarded.
    void backward(const py::object& y) {
        require_built();
        std::vector<float> t = check_target(y);
        net_->zeroAllGradients();
        net_->accumulateGradients(t);
        net_->applyGradients();
    }

    double loss(const py::object& x, const py::object& y) {
        require_built();
        std::vector<float> t = check_target(y);
        forward(x);
        return static_cast<double>(net_->computeLoss(t));
    }

    // -------- training, entirely inside the engine

    std::vector<float> train(const py::object& X, const py::object& Y, int epochs,
                             bool shuffle, std::optional<unsigned> seed,
                             bool verbose, int log_every) {
        require_built();
        if (epochs < 0) throw std::invalid_argument("epochs must be >= 0");
        auto xs = to_mat(X, "X");
        auto ys = to_mat(Y, "Y");
        if (xs.size() != ys.size())
            throw std::invalid_argument("X and Y must have the same number of rows (got " +
                                        std::to_string(xs.size()) + " and " +
                                        std::to_string(ys.size()) + ")");
        if (xs.empty() || epochs == 0) return {};

        // The engine indexes rows directly, so a malformed row would read out of
        // bounds rather than raise. Check every row up front.
        const size_t n_in = static_cast<size_t>(specs_.front().in);
        const size_t n_out = static_cast<size_t>(output_layer()->layerWidth);
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i].size() != n_in)
                throw std::invalid_argument("X row " + std::to_string(i) + " has " +
                    std::to_string(xs[i].size()) + " features, expected " + std::to_string(n_in));
            if (ys[i].size() != n_out)
                throw std::invalid_argument("Y row " + std::to_string(i) + " has " +
                    std::to_string(ys[i].size()) + " targets, expected " + std::to_string(n_out));
        }
        if (log_every <= 0) log_every = std::max(1, epochs / 10);

        std::vector<float> history(static_cast<size_t>(epochs));
        {
            // The engine touches no Python objects, so other threads may run.
            py::gil_scoped_release unlock;
            net_->train(xs, ys, epochs, history.data(), epochs, verbose, shuffle, seed, log_every);
            if (verbose) std::cout.flush();
        }
        return history;
    }

    // One engine batch step over the given samples: zero, accumulate, scale, apply.
    double train_batch(const py::object& X, const py::object& Y) {
        require_built();
        auto xs = to_mat(X, "X");
        auto ys = to_mat(Y, "Y");
        if (xs.size() != ys.size())
            throw std::invalid_argument("X and Y must have the same number of rows");
        if (xs.empty()) throw std::invalid_argument("train_batch needs at least one sample");
        const size_t n_in = static_cast<size_t>(specs_.front().in);
        const size_t n_out = static_cast<size_t>(output_layer()->layerWidth);
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i].size() != n_in || ys[i].size() != n_out)
                throw std::invalid_argument("row " + std::to_string(i) + " has the wrong shape; "
                    "expected " + std::to_string(n_in) + " features and " +
                    std::to_string(n_out) + " targets");
        }
        py::gil_scoped_release unlock;
        return static_cast<double>(
            net_->trainBatch(0, static_cast<int>(xs.size()), xs, ys));
    }

    // Mean loss over a whole dataset, measured in the engine. A Python loop
    // calling loss() per row costs a round trip per sample; this is one call,
    // which is what makes a 625-point loss-landscape sweep practical.
    double evaluate(const py::object& X, const py::object& Y) {
        require_built();
        auto xs = to_mat(X, "X");
        auto ys = to_mat(Y, "Y");
        if (xs.size() != ys.size())
            throw std::invalid_argument("X and Y must have the same number of rows (got " +
                                        std::to_string(xs.size()) + " and " +
                                        std::to_string(ys.size()) + ")");
        if (xs.empty()) throw std::invalid_argument("evaluate needs at least one sample");

        const size_t n_in = static_cast<size_t>(specs_.front().in);
        const size_t n_out = static_cast<size_t>(output_layer()->layerWidth);
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i].size() != n_in)
                throw std::invalid_argument("X row " + std::to_string(i) + " has " +
                    std::to_string(xs[i].size()) + " features, expected " + std::to_string(n_in));
            if (ys[i].size() != n_out)
                throw std::invalid_argument("Y row " + std::to_string(i) + " has " +
                    std::to_string(ys[i].size()) + " targets, expected " + std::to_string(n_out));
        }

        py::gil_scoped_release unlock;
        double total = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) {
            net_->forwardPass(xs[i].data());
            total += static_cast<double>(net_->computeLoss(ys[i]));
        }
        return total / static_cast<double>(xs.size());
    }

    // Overwrite a layer's parameters. Needed to walk the network away from its
    // trained point and back again (loss landscapes), and to restore a model
    // whose weights were saved elsewhere.
    void set_weights(int layer, const py::object& values) {
        core::Layer* l = layer_at(layer);
        Arr a = Arr::ensure(values);
        if (!a) throw std::invalid_argument("weights must be an array of numbers");
        if (a.ndim() != 1 && a.ndim() != 2)
            throw std::invalid_argument("weights must be 1-D or 2-D, got " +
                                        std::to_string(a.ndim()) + "-D");
        const size_t want = static_cast<size_t>(l->layerWidth) * static_cast<size_t>(l->inputSize);
        if (static_cast<size_t>(a.size()) != want)
            throw std::invalid_argument(
                "layer " + std::to_string(layer) + " holds " + std::to_string(want) +
                " weights (" + std::to_string(l->layerWidth) + "x" +
                std::to_string(l->inputSize) + "), got " + std::to_string(a.size()));
        std::copy(a.data(), a.data() + a.size(), l->neuronWeights.begin());
    }

    void set_biases(int layer, const py::object& values) {
        core::Layer* l = layer_at(layer);
        std::vector<float> b = to_vec(values, "biases");
        if (b.size() != static_cast<size_t>(l->layerWidth))
            throw std::invalid_argument(
                "layer " + std::to_string(layer) + " holds " +
                std::to_string(l->layerWidth) + " biases, got " + std::to_string(b.size()));
        std::copy(b.begin(), b.end(), l->neuronBiases.begin());
    }

    // -------- introspection

    int num_hidden() const { return built_ ? net_->numHiddenLayers
                                           : static_cast<int>(count_hidden()); }
    int input_size() const {
        if (specs_.empty()) throw std::runtime_error("no layers added yet");
        return specs_.front().in;
    }
    int output_size() const { require_built(); return output_layer()->layerWidth; }

    // One entry per layer, in forward order. plotting.py draws the architecture
    // from this rather than re-deriving shapes from the weight matrices.
    py::list layer_specs() const {
        py::list out;
        for (const auto& s : specs_) {
            py::dict d;
            d["kind"]       = s.is_output ? "output" : "hidden";
            d["width"]      = s.width;
            d["input_size"] = s.in;
            d["activation"] = s.act_name;
            d["init"]       = s.init_name;
            if (s.is_output) d["loss"] = s.loss_name;
            out.append(d);
        }
        return out;
    }

    py::array_t<float> weights(int layer) const {
        const core::Layer* l = layer_at(layer);
        py::array_t<float> out({l->layerWidth, l->inputSize});
        std::copy(l->neuronWeights.begin(), l->neuronWeights.end(), out.mutable_data());
        return out;
    }
    Arr biases(int layer) const { return to_numpy(layer_at(layer)->neuronBiases); }

    double learning_rate() const {
        return static_cast<double>(built_ ? net_->learningRate : lr_);
    }
    void set_learning_rate(double v) {
        if (!(v > 0.0)) throw std::invalid_argument("learning_rate must be > 0");
        lr_ = static_cast<float>(v);
        if (built_) net_->learningRate = lr_;
    }
    int batch_size() const { return built_ ? net_->batches : batch_; }
    void set_batch_size(int v) {
        if (v < 0) throw std::invalid_argument("batch_size must be >= 0 (0 means one batch per epoch)");
        batch_ = v;
        if (built_) net_->batches = v;
    }
    bool is_built() const { return built_; }

    std::string repr() const {
        std::ostringstream os;
        os << "<gradiex.Network lr=" << learning_rate() << " batch=" << batch_size();
        if (specs_.empty()) { os << " (no layers)>"; return os.str(); }
        os << " " << specs_.front().in;
        for (const auto& s : specs_) os << " -> " << s.width << "[" << s.act_name << "]";
        if (has_output()) os << " loss=" << specs_.back().loss_name;
        os << (built_ ? "" : " (not built)") << ">";
        return os.str();
    }

private:
    struct Spec {
        int width = 0, in = 0;
        act::ActivationFunction a = nullptr;
        los::LossFunction l = nullptr;
        ini::initFunction init = nullptr;
        bool is_output = false;
        std::string act_name, loss_name, init_name;
    };

    static int require_positive(int v, const char* what) {
        if (v <= 0) throw std::invalid_argument(std::string(what) + " must be > 0, got " +
                                                std::to_string(v));
        return v;
    }
    int resolve_input_size(std::optional<int> given) const {
        if (given.has_value()) {
            int v = require_positive(*given, "input_size");
            if (!specs_.empty() && v != specs_.back().width)
                throw std::invalid_argument(
                    "input_size=" + std::to_string(v) + " does not match the previous layer's "
                    "width of " + std::to_string(specs_.back().width));
            return v;
        }
        if (specs_.empty())
            throw std::invalid_argument("input_size is required for the first layer");
        return specs_.back().width;
    }
    size_t count_hidden() const {
        size_t n = 0; for (const auto& s : specs_) if (!s.is_output) ++n; return n;
    }
    bool has_output() const { return !specs_.empty() && specs_.back().is_output; }
    void require_built() const {
        if (!built_) throw std::runtime_error("call build() before using the network");
    }
    core::OutputLayer* output_layer() const { return net_->outputLayers[0]; }

    core::Layer* layer_at(int i) const {
        require_built();
        int nh = net_->numHiddenLayers;
        if (i < 0) i += nh + 1;
        if (i < 0 || i > nh)
            throw std::out_of_range("layer index out of range: " + std::to_string(i) +
                                    " (network has " + std::to_string(nh) + " hidden + 1 output)");
        if (i == nh) return net_->outputLayers[0];
        return net_->hiddenLayers[static_cast<size_t>(i)];
    }
    std::vector<float> check_target(const py::object& y) const {
        std::vector<float> t = to_vec(y, "y");
        check_len(t.size(), static_cast<size_t>(output_layer()->layerWidth), "y", "the output width");
        return t;
    }
    static void check_len(size_t got, size_t want, const char* what, const char* against) {
        if (got != want)
            throw std::invalid_argument(std::string(what) + " has length " + std::to_string(got) +
                                        " but " + against + " is " + std::to_string(want));
    }

    std::vector<Spec> specs_;
    std::unique_ptr<core::Network> net_;
    float lr_ = 0.01f;
    int batch_ = 1;
    bool built_ = false;
};

// -------------------------------------------------------------------- module

// The extension is built as gradiex._core; gradiex/__init__.py re-exports it
// and layers the matplotlib plotting on top.
PYBIND11_MODULE(_core, m) {
    m.doc() = "GradieX -- a dependency-free feedforward neural network engine in C++.";
    m.attr("__version__") = py::str("0.0.3");
    m.attr("__author__") = py::str("Snehashish Laskar");
    m.attr("__dtype__") = py::str("float32");

    m.def("seed", [](unsigned int value) { ini::setSeed(value); }, py::arg("value"),
          "Seed the engine's weight-initialisation RNG, so build() is reproducible. "
          "This is separate from train(seed=...), which only seeds the shuffle.");

    m.def("activations", []{
        std::vector<std::string> v;
        for (const auto& kv : activations()) v.push_back(kv.first);
        return v;
    }, "Names accepted by the `activation` argument.");
    m.def("losses", []{
        std::vector<std::string> v;
        for (const auto& kv : losses()) v.push_back(kv.first);
        return v;
    }, "Names accepted by the `loss` argument.");
    m.def("initializers", []{
        std::vector<std::string> v;
        for (const auto& kv : initializers()) v.push_back(kv.first);
        return v;
    }, "Names accepted by the `init` argument.");

    py::class_<Network>(m, "Network", R"doc(
A feedforward network.

Layers are declared, then built:

    net = gradiex.Network(learning_rate=0.5, batch_size=4)
    net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
    net.add_output(1, activation="sigmoid", loss="mse")
    net.build()

    history = net.train(X, Y, epochs=4000, shuffle=True, seed=42)
    net.predict([0.1, 0.2])

train() runs entirely inside the C++ engine: the epoch loop, the shuffle, the
mini-batching, gradient accumulation and the SGD update are all engine code, so
training from Python is the same code path as training from main.cpp. The engine
is float32 throughout; float64 input is accepted and cast on the way in.

batch_size is the number of samples whose gradients are accumulated before one
update: 1 is per-sample SGD, 0 means one batch per epoch (full batch).
)doc")
        .def(py::init<double, int>(), py::arg("learning_rate") = 0.01,
             py::arg("batch_size") = 1)
        .def("add_hidden", &Network::add_hidden,
             py::arg("width"), py::arg("input_size") = std::nullopt,
             py::arg("activation") = "relu", py::arg("init") = "he",
             "Append a hidden layer. input_size defaults to the previous layer's width.")
        .def("add_output", &Network::add_output,
             py::arg("width"), py::arg("input_size") = std::nullopt,
             py::arg("activation") = "sigmoid", py::arg("loss") = "mse",
             py::arg("init") = "he",
             "Add the output layer. Must be called once, after all hidden layers.")
        .def("build", &Network::build, "Construct the engine and initialize weights.")
        .def("initialize", &Network::build, "Alias for build().")
        .def("forward", &Network::forward, py::arg("x"),
             "Run one forward pass and return the output activations.")
        .def("predict", &Network::forward, py::arg("x"), "Alias for forward().")
        .def("backward", &Network::backward, py::arg("y"),
             "Backpropagate one target and apply an SGD update. Call forward() first.")
        .def("zero_gradients", &Network::zero_gradients,
             "Clear the accumulated gradients of every layer.")
        .def("accumulate", &Network::accumulate, py::arg("y"),
             "Accumulate one sample's gradients into every layer. Call forward() first.")
        .def("scale_gradients", &Network::scale_gradients, py::arg("factor"),
             "Scale every accumulated gradient, e.g. by 1/batch_size before applying.")
        .def("apply_gradients", &Network::apply_gradients,
             "Apply the accumulated gradients as one SGD step.")
        .def("loss", &Network::loss, py::arg("x"), py::arg("y"),
             "Forward x, then return the configured loss against y.")
        .def("train", &Network::train,
             py::arg("X"), py::arg("Y"), py::arg("epochs") = 100,
             py::arg("shuffle") = false, py::arg("seed") = std::nullopt,
             py::arg("verbose") = false, py::arg("log_every") = 0,
             "Train in the engine. Returns the mean loss of each epoch.")
        .def("train_batch", &Network::train_batch, py::arg("X"), py::arg("Y"),
             "Run one engine batch step over these samples. Returns the mean batch loss.")
        .def("evaluate", &Network::evaluate, py::arg("X"), py::arg("Y"),
             "Mean loss over a dataset, computed in the engine. No update is applied.")
        .def("weights", &Network::weights, py::arg("layer"),
             "Weights of a layer as a (width, input_size) array. Negative indices allowed.")
        .def("biases", &Network::biases, py::arg("layer"), "Biases of a layer.")
        .def("set_weights", &Network::set_weights, py::arg("layer"), py::arg("values"),
             "Overwrite a layer's weights. Accepts (width, input_size) or flat.")
        .def("set_biases", &Network::set_biases, py::arg("layer"), py::arg("values"),
             "Overwrite a layer's biases.")
        .def_property("learning_rate", &Network::learning_rate, &Network::set_learning_rate)
        .def_property("batch_size", &Network::batch_size, &Network::set_batch_size)
        .def_property_readonly("num_hidden", &Network::num_hidden)
        .def_property_readonly("input_size", &Network::input_size)
        .def_property_readonly("output_size", &Network::output_size)
        .def_property_readonly("built", &Network::is_built)
        .def_property_readonly("layers", &Network::layer_specs,
             "One dict per layer in forward order: kind, width, input_size, "
             "activation, init, and loss on the output layer.")
        .def("__repr__", &Network::repr);
}
