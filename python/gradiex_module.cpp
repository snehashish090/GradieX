// Python bindings for the GradieX v2 engine.
//
// Design note: the C++ API lets a caller reach several states that crash or
// silently train wrong -- an out-of-range layer index, unfilled layer slots,
// a mismatched activation/derivative pair, softmax with no derivative, more
// than one output layer (backwardPass only propagates from outputLayers[0]).
// This layer collects layer specs first and constructs core::Network only in
// build(), so none of those states are reachable from Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/network.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace act = core::functions::activations;
namespace ini = core::functions::initializers;
namespace los = core::functions::loss;

// ---------------------------------------------------------------- name tables

struct ActPair {
    act::ActivationFunction f;
    act::ActivationFunctionDerivative d;   // null when no derivative exists yet
};
struct LossPair {
    los::LossFunction f;
    los::LossFunctionDerivative d;
};

static const std::map<std::string, ActPair>& activations() {
    static const std::map<std::string, ActPair> t{
        {"sigmoid",    {act::sigmoid,   act::sigmoidDerivative}},
        {"relu",       {act::relu,      act::reluDerivative}},
        {"leaky_relu", {act::leakyRelu, act::leakyReluDerivative}},
        {"tanh",       {act::tanh,      act::tanhDerivative}},
        {"swish",      {act::swish,     act::swishDerivative}},
        {"gelu",       {act::gelu,      act::geluDerivative}},
        {"softmax",    {act::softmax,   nullptr}},
    };
    return t;
}
static const std::map<std::string, LossPair>& losses() {
    static const std::map<std::string, LossPair> t{
        {"mse",                  {los::mse,    los::mseDerivative}},
        {"mae",                  {los::mae,    los::maeDerivative}},
        {"huber",                {los::huber,  los::huberDerivative}},
        {"binary_cross_entropy", {los::binaryCrossEntropy,
                                  los::binaryCrossEntropyDerivative}},
        {"cross_entropy",        {los::crossEntropy, los::crossEntropyDerivative}},
        {"bce",                  {los::binaryCrossEntropy,
                                  los::binaryCrossEntropyDerivative}},
        {"cce",                  {los::crossEntropy, los::crossEntropyDerivative}},
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

static ActPair lookup_activation(const std::string& name, const char* where) {
    auto it = activations().find(name);
    if (it == activations().end())
        throw std::invalid_argument("unknown activation '" + name + "' for " + where +
                                    "; available: " + known_keys(activations()));
    if (it->second.d == nullptr)
        throw std::invalid_argument(
            "activation '" + name + "' has no derivative implemented in the engine yet, so it "
            "cannot be trained through. The softmax backward path needs to be fused with "
            "categorical cross-entropy (dL/dz = p - y); that is not implemented.");
    return it->second;
}
static LossPair lookup_loss(const std::string& name) {
    auto it = losses().find(name);
    if (it == losses().end())
        throw std::invalid_argument("unknown loss '" + name + "'; available: " + known_keys(losses()));
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

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<double> to_vec(const py::object& o, const char* name) {
    Arr a = Arr::ensure(o);
    if (!a) throw std::invalid_argument(std::string(name) + " must be a sequence of numbers");
    if (a.ndim() != 1)
        throw std::invalid_argument(std::string(name) + " must be 1-D, got " +
                                    std::to_string(a.ndim()) + "-D");
    const double* p = a.data();
    return std::vector<double>(p, p + a.size());
}

static std::vector<std::vector<double>> to_mat(const py::object& o, const char* name) {
    Arr a = Arr::ensure(o);
    if (!a) throw std::invalid_argument(std::string(name) + " must be a 2-D array of numbers");
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(name) + " must be 2-D (n_samples, n_features), got " +
                                    std::to_string(a.ndim()) + "-D");
    auto r = a.unchecked<2>();
    std::vector<std::vector<double>> out(static_cast<size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<size_t>(i)].resize(static_cast<size_t>(r.shape(1)));
        for (py::ssize_t j = 0; j < r.shape(1); ++j)
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = r(i, j);
    }
    return out;
}

static Arr to_numpy(const std::vector<double>& v) {
    Arr out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// ------------------------------------------------------------------- Network

class Network {
public:
    explicit Network(double learning_rate) : lr_(learning_rate) {
        if (!(learning_rate > 0.0))
            throw std::invalid_argument("learning_rate must be > 0");
    }
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    void add_hidden(int width, std::optional<int> input_size,
                    const std::string& activation, const std::string& init) {
        if (built_) throw std::runtime_error("cannot add layers after build()");
        if (!specs_.empty() && specs_.back().is_output)
            throw std::runtime_error("hidden layers must be added before the output layer");
        Spec s;
        s.width = require_positive(width, "width");
        s.in = resolve_input_size(input_size);
        s.a = lookup_activation(activation, "a hidden layer");
        s.init = lookup_init(init);
        s.l = lookup_loss("mse");   // unused on hidden layers; the engine stores one anyway
        s.is_output = false;
        s.act_name = activation;
        specs_.push_back(s);
    }

    void add_output(int width, std::optional<int> input_size,
                    const std::string& activation, const std::string& loss,
                    const std::string& init) {
        if (built_) throw std::runtime_error("cannot add layers after build()");
        if (has_output())
            throw std::runtime_error(
                "only one output layer is supported: the engine's backwardPass propagates "
                "gradients from outputLayers[0] only, so additional output layers would "
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
        specs_.push_back(s);
    }

    void build() {
        if (built_) throw std::runtime_error("build() has already been called");
        if (!has_output())
            throw std::runtime_error("add an output layer with add_output() before build()");
        int n_hidden = 0;
        for (const auto& s : specs_) if (!s.is_output) ++n_hidden;

        net_ = std::make_unique<core::Network>(n_hidden, 1, lr_);
        int idx = 0;
        for (const auto& s : specs_) {
            if (s.is_output)
                net_->addOutputLayer(0, s.width, s.in, s.a.f, s.a.d, s.init, s.l.f, s.l.d);
            else
                net_->addHiddenLayer(idx++, s.width, s.in, s.a.f, s.a.d, s.init, s.l.f, s.l.d);
        }
        net_->initializeAllWeightsAndBiases();
        built_ = true;
    }

    Arr forward(const py::object& x) {
        require_built();
        std::vector<double> in = to_vec(x, "x");
        check_len(in.size(), static_cast<size_t>(specs_.front().in), "x", "the first layer's input_size");
        net_->forwardPass(in.data());
        return to_numpy(output_layer()->neuronOutputsActivated);
    }

    void backward(const py::object& y) {
        require_built();
        std::vector<double> t = to_vec(y, "y");
        check_len(t.size(), static_cast<size_t>(output_layer()->layerWidth), "y", "the output width");
        net_->backwardPass(t);
    }

    double loss(const py::object& x, const py::object& y) {
        require_built();
        std::vector<double> t = to_vec(y, "y");
        check_len(t.size(), static_cast<size_t>(output_layer()->layerWidth), "y", "the output width");
        forward(x);
        return eval_loss(t);
    }

    // Returns mean loss per epoch, computed with the layer's configured loss
    // function. (core::Network::computeLoss hardcodes a squared-error sum
    // regardless of the configured loss; this does not.)
    std::vector<double> train(const py::object& X, const py::object& Y, int epochs,
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
        std::vector<double> history;
        if (xs.empty() || epochs == 0) return history;

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

        std::vector<size_t> order(xs.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::mt19937 rng(seed.has_value() ? *seed : std::random_device{}());
        if (log_every <= 0) log_every = std::max(1, epochs / 10);

        history.reserve(static_cast<size_t>(epochs));
        {
            py::gil_scoped_release unlock;
            for (int e = 0; e < epochs; ++e) {
                if (shuffle) std::shuffle(order.begin(), order.end(), rng);
                double sum = 0.0;
                for (size_t k : order) {
                    net_->forwardPass(xs[k].data());
                    sum += eval_loss(ys[k]);
                    std::vector<double> t = ys[k];       // backwardPass takes a mutable ref
                    net_->backwardPass(t);
                }
                double mean = sum / static_cast<double>(xs.size());
                history.push_back(mean);
                if (verbose && (e % log_every == 0 || e == epochs - 1))
                    std::cout << "epoch " << e << "  loss " << mean << "\n";
            }
            if (verbose) std::cout.flush();
        }
        return history;
    }

    // -------- introspection

    int num_hidden() const { return built_ ? net_->numHiddenLayers
                                           : static_cast<int>(count_hidden()); }
    int input_size() const {
        if (specs_.empty()) throw std::runtime_error("no layers added yet");
        return specs_.front().in;
    }
    int output_size() const { require_built_const(); return output_layer()->layerWidth; }

    py::array_t<double> weights(int layer) const {
        const core::Layer* l = layer_at(layer);
        py::array_t<double> out({l->layerWidth, l->inputSize});
        std::copy(l->neuronWeights.begin(), l->neuronWeights.end(), out.mutable_data());
        return out;
    }
    Arr biases(int layer) const { return to_numpy(layer_at(layer)->neuronBiases); }

    double learning_rate() const { return built_ ? net_->learningRate : lr_; }
    void set_learning_rate(double v) {
        if (!(v > 0.0)) throw std::invalid_argument("learning_rate must be > 0");
        lr_ = v;
        if (built_) net_->learningRate = v;
    }
    bool is_built() const { return built_; }

    std::string repr() const {
        std::ostringstream os;
        os << "<gradiex.Network lr=" << learning_rate();
        if (specs_.empty()) { os << " (no layers)>"; return os.str(); }
        os << " " << specs_.front().in;
        for (const auto& s : specs_) os << " -> " << s.width << "[" << s.act_name << "]";
        if (has_output()) os << " loss=" << output_spec().loss_name;
        os << (built_ ? "" : " (not built)") << ">";
        return os.str();
    }

private:
    struct Spec {
        int width = 0, in = 0;
        ActPair a{};
        LossPair l{};
        ini::initFunction init = nullptr;
        bool is_output = false;
        std::string act_name, loss_name;
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
    bool has_output() const {
        return !specs_.empty() && specs_.back().is_output;
    }
    const Spec& output_spec() const { return specs_.back(); }
    void require_built() const {
        if (!built_) throw std::runtime_error("call build() before using the network");
    }
    void require_built_const() const { require_built(); }
    core::OutputLayer* output_layer() const { return net_->outputLayers[0]; }

    const core::Layer* layer_at(int i) const {
        require_built();
        int nh = net_->numHiddenLayers;
        if (i < 0) i += nh + 1;
        if (i < 0 || i > nh)
            throw std::out_of_range("layer index out of range: " + std::to_string(i) +
                                    " (network has " + std::to_string(nh) + " hidden + 1 output)");
        if (i == nh) return net_->outputLayers[0];
        return net_->hiddenLayers[static_cast<size_t>(i)];
    }
    double eval_loss(const std::vector<double>& target) const {
        core::OutputLayer* o = output_layer();
        return o->lossFunction(target.data(), o->neuronOutputsActivated.data(),
                               static_cast<size_t>(o->layerWidth));
    }
    static void check_len(size_t got, size_t want, const char* what, const char* against) {
        if (got != want)
            throw std::invalid_argument(std::string(what) + " has length " + std::to_string(got) +
                                        " but " + against + " is " + std::to_string(want));
    }

    std::vector<Spec> specs_;
    std::unique_ptr<core::Network> net_;
    double lr_;
    bool built_ = false;
};

// -------------------------------------------------------------------- module

PYBIND11_MODULE(gradiex, m) {
    m.doc() = "GradieX -- a dependency-free feedforward neural network engine in C++.";
    m.attr("__version__") = py::str("0.0.1");
    m.attr("__author__") = py::str("Snehashish Laskar");

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

    net = gradiex.Network(learning_rate=0.05)
    net.add_hidden(4, input_size=2, activation="sigmoid", init="xavier")
    net.add_output(1, activation="sigmoid", loss="mse")
    net.build()

    history = net.train(X, Y, epochs=2000, shuffle=True, seed=42)
    net.predict([0.1, 0.2])
)doc")
        .def(py::init<double>(), py::arg("learning_rate") = 0.01)
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
        .def("loss", &Network::loss, py::arg("x"), py::arg("y"),
             "Forward x, then return the configured loss against y.")
        .def("train", &Network::train,
             py::arg("X"), py::arg("Y"), py::arg("epochs") = 100,
             py::arg("shuffle") = false, py::arg("seed") = std::nullopt,
             py::arg("verbose") = false, py::arg("log_every") = 0,
             "Train by per-sample SGD. Returns mean loss per epoch.")
        .def("weights", &Network::weights, py::arg("layer"),
             "Weights of a layer as a (width, input_size) array. Negative indices allowed.")
        .def("biases", &Network::biases, py::arg("layer"), "Biases of a layer.")
        .def_property("learning_rate", &Network::learning_rate, &Network::set_learning_rate)
        .def_property_readonly("num_hidden", &Network::num_hidden)
        .def_property_readonly("input_size", &Network::input_size)
        .def_property_readonly("output_size", &Network::output_size)
        .def_property_readonly("built", &Network::is_built)
        .def("__repr__", &Network::repr);
}
