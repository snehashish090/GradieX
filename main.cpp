/*
=============================================================================
                                                                            =
      ____  ____      _     ____   ___  _____ __  __                        =
     / ___||  _ \    / \   |  _ \ |_ _|| ____|\ \/ /                        =
    | |  _ | |_) |  / _ \  | | | | | | |  _|   \  /                         =
    | |_| ||  _ <  / ___ \ | |_| | | | | |___  /  \                         =
     \____||_| \_\/_/   \_\|____/ |___||_____|/_/\_\                        =
                                                                            =
    A neural network library in C++17                                       =
                                                                            =
=============================================================================
    Version   : 0.1.0
    Author    : Snehashish Laskar
    License   : MIT (see LICENSE)
    Repository: https://github.com/<owner>/GradieX
=============================================================================

Gradiex is a Neural Network Library written in C++ including various activation
and loss functions. The program runs through both forward and backward passes
to calculate for the loss gradient and update its internal weights.

It is a single translation unit with no dependencies beyond the C++17 standard
library. A trained model can be saved to disk and loaded back into any C++
program, which makes it usable for inference in places where a Python runtime
is not an option.

-----------------------------------------------------------------------------
                                CODE ARCHIECTURE
-----------------------------------------------------------------------------

    Activation / Loss        : enums, plus parse_* and *_name helpers
    Neuron                   : weights, bias, caches, gradient accumulators
    Layer                    : a vector of Neurons + the layer-wide activation
    TrainOptions             : knobs for a training run
    TrainHistory             : per-epoch losses returned by train()
    NeuralNetwork            : the network, the training loop, save / load
    main()                   : two worked examples (XOR, three-class spirals)

-----------------------------------------------------------------------------
         IN THIS DOCUMENTATION WE REFER TO VECTORS IN THE FORMAT :v:
-----------------------------------------------------------------------------

    :z:   pre-activations of a layer   (z_i = w_i . input + b_i)
    :a:   activations, f(:z:)          (what the next layer receives)
    :p:   the network's predictions    (:a: of the output layer)
    :y:   the target / ground truth
    n     the number of elements in the vector under discussion

Activation functions:
    - Sigmoid : f(x) = 1 / (1 + e^-x)
    - TanH : f(x) = e^x - e^-x / e^x + e^-x
    - ReLU : f(x) = max(0,x)
    - linear : f(x) = x
    - leaky ReLU : f(x) = x if x > 0 else a*x        (a = 0.01, so max(a*x, x))
    - softmax : f(:z:)_i = e^(:z:_i) / sum(e^(:z:_j) for j = 1..n)

    Sigmoid is evaluated by branching on the sign of x, so neither e^-x nor
    e^x can overflow. Softmax subtracts max(:z:) before exponentiating, for
    the same reason; both are stable out to +/-1e5.

Activation Derivatives (Backward Pass):
    - d_Sigmoid : f'(x) = f(x) * (1 - f(x))
    - d_TanH : f'(x) = 1 - f(x)^2
    - d_ReLU : f'(x) = 1 if x > 0 else 0
    - d_linear : f'(x) = 1
    - d_leaky ReLU : f'(x) = 1 if x > 0 else a
    - d_softmax : f'(:z:) = p_i * (delta_ij - p_j)

    Note on d_softmax: every other derivative above is elementwise, so it is
    a scalar-in / scalar-out function. Softmax's Jacobian is a dense n-by-n
    matrix, which that signature cannot express. It is therefore never
    evaluated on its own - it is fused with Categorical Cross Entropy (see
    DATA FLOW: BACKWARD). Neuron::activation_derivative throws if asked for it.

Loss Functions:
    - Mean Squared Error : f(x) = (1/n)*sum[i=1, n]((y_i - :z:_i)^2)
    - Binary Cross Entropy : f(x) = -(1/n)*sum[i=1, n](y_i * log(:p:_i) +
                                    (1 - y_i) * log(1 - :p:_i))
    - Categorical Cross Entropy : f(x) = -sum[i=1, n](y_i * log(:p:_i))

    MSE and BCE average over the n outputs; CCE sums over the n classes,
    which is the standard definition for a one-hot target. Both cross
    entropies clamp :p: away from 0 and 1 by epsilon = 1e-9, so a fully
    confident wrong answer costs a large finite number rather than infinity.

Loss Derivatives (Backward Pass):
    - d_Mean Squared Error : f'(x) = (2/n) * (:p:_i - y_i)
    - d_Binary Cross Entropy : f'(x) = (:p:_i - y_i) /
                                       (:p:_i * (1 - :p:_i) * n)
    - d_Categorical Cross Entropy : f'(x) = -y_i / :p:_i

    d_CCE above is dL/d:p:. It is only used if CCE is somehow paired with a
    non-softmax output; the supported path never evaluates it, because the
    fused form below avoids the division by :p:_i entirely.
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// Activation and loss are enums so the hot path switches on an int instead of
// comparing strings per neuron per pass. The string spellings stay available
// at the API boundary, where they are parsed exactly once.
// ---------------------------------------------------------------------------

enum class Activation { Sigmoid, Tanh, ReLU, Linear, LeakyReLU, Softmax };
enum class Loss { MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy };

inline Activation parse_activation(const string& name) {
    if (name == "sigmoid") return Activation::Sigmoid;
    if (name == "tanh") return Activation::Tanh;
    if (name == "ReLU") return Activation::ReLU;
    if (name == "linear") return Activation::Linear;
    if (name == "leaky_ReLU") return Activation::LeakyReLU;
    if (name == "softmax") return Activation::Softmax;
    throw invalid_argument("Unsupported activation function: " + name);
}

inline string activation_name(Activation activation) {
    switch (activation) {
        case Activation::Sigmoid: return "sigmoid";
        case Activation::Tanh: return "tanh";
        case Activation::ReLU: return "ReLU";
        case Activation::Linear: return "linear";
        case Activation::LeakyReLU: return "leaky_ReLU";
        case Activation::Softmax: return "softmax";
    }
    throw invalid_argument("Unknown activation enum value.");
}

inline Loss parse_loss(const string& name) {
    if (name == "mean_squared_error") return Loss::MeanSquaredError;
    if (name == "binary_cross_entropy") return Loss::BinaryCrossEntropy;
    if (name == "categorical_cross_entropy") return Loss::CategoricalCrossEntropy;
    throw invalid_argument("Unsupported loss function: " + name);
}

inline string loss_name(Loss loss) {
    switch (loss) {
        case Loss::MeanSquaredError: return "mean_squared_error";
        case Loss::BinaryCrossEntropy: return "binary_cross_entropy";
        case Loss::CategoricalCrossEntropy: return "categorical_cross_entropy";
    }
    throw invalid_argument("Unknown loss enum value.");
}

/*
-----------------------------------------------------------------------------
                            ARCHITECTURE
-----------------------------------------------------------------------------

    Neuron          one weight vector, one bias. Caches the input it last
                    saw, its pre-activation :z:_i, and its activation.
                    Also holds this batch's accumulated gradients.

    Layer           a vector of Neurons sharing one activation. Owns the
                    layer-wide steps: softmax (which needs every :z: before
                    any :a: can be computed) and the dropout mask.

    NeuralNetwork   hidden_layers + one output_layer, a loss, and the RNG.
                    Chains the layers, owns the training loop, and handles
                    persistence.

    Widths: every hidden layer shares one width, fixed by the constructor.
    load() reads each layer's shape independently, so a hand-written model
    file with varying widths will load and train correctly - the uniform
    width is a constructor limitation, not an engine one.
*/

class Neuron {
public:
    vector<double> weights;
    double bias;
    vector<double> input_cache;
    double pre_activation;
    double output;              // activation output BEFORE any dropout mask

    // Gradient accumulators, so a batch can sum before a single update.
    vector<double> weight_gradients;
    double bias_gradient;

    Neuron() : bias(0.0), pre_activation(0.0), output(0.0), bias_gradient(0.0) {}

    Neuron(int input_size, mt19937& rng)
        : bias(0.0), pre_activation(0.0), output(0.0), bias_gradient(0.0) {
        normal_distribution<double> dist(0.0, sqrt(2.0 / static_cast<double>(max(1, input_size))));
        weights.resize(input_size);
        for (double& weight : weights) {
            weight = dist(rng);
        }
        weight_gradients.assign(static_cast<size_t>(max(0, input_size)), 0.0);
    }

    static double activate(double value, Activation activation) {
        switch (activation) {
            case Activation::Sigmoid:
                if (value >= 0.0) {
                    return 1.0 / (1.0 + exp(-value));
                } else {
                    double exp_value = exp(value);
                    return exp_value / (1.0 + exp_value);
                }
            case Activation::Tanh: return tanh(value);
            case Activation::ReLU: return max(0.0, value);
            case Activation::Linear: return value;
            case Activation::LeakyReLU: return value > 0.0 ? value : 0.01 * value;
            case Activation::Softmax:
                throw invalid_argument("Softmax is a layer-wide activation; use Layer::forward.");
        }
        throw invalid_argument("Unknown activation enum value.");
    }

    static double activation_derivative(double pre_activation_value, double output_value, Activation activation) {
        switch (activation) {
            case Activation::Sigmoid: return output_value * (1.0 - output_value);
            case Activation::Tanh: return 1.0 - (output_value * output_value);
            case Activation::ReLU: return pre_activation_value > 0.0 ? 1.0 : 0.0;
            case Activation::Linear: return 1.0;
            case Activation::LeakyReLU: return pre_activation_value > 0.0 ? 1.0 : 0.01;
            case Activation::Softmax:
                throw invalid_argument("Softmax has a non-diagonal Jacobian; it is fused with the loss.");
        }
        throw invalid_argument("Unknown activation enum value.");
    }

    // String-accepting wrappers, kept so existing call sites and examples work.
    static double activate(double value, const string& activation) {
        return activate(value, parse_activation(activation));
    }
    static double activation_derivative(double pre_activation_value, double output_value, const string& activation) {
        return activation_derivative(pre_activation_value, output_value, parse_activation(activation));
    }

    void compute_pre_activation(const vector<double>& input) {
        if (input.size() != weights.size()) {
            throw invalid_argument("Input size does not match neuron weight size.");
        }
        input_cache = input;
        pre_activation = bias;
        for (size_t index = 0; index < input.size(); ++index) {
            pre_activation += weights[index] * input[index];
        }
    }

    double forward(const vector<double>& input, Activation activation) {
        compute_pre_activation(input);
        output = activate(pre_activation, activation);
        return output;
    }

    double forward(const vector<double>& input, const string& activation) {
        return forward(input, parse_activation(activation));
    }

    void zero_gradients() {
        weight_gradients.assign(weights.size(), 0.0);
        bias_gradient = 0.0;
    }

    // scale averages the accumulated batch; l2 shrinks weights (never the bias).
    void apply_gradients(double learning_rate, double scale, double l2) {
        for (size_t index = 0; index < weights.size(); ++index) {
            double gradient = weight_gradients[index] * scale + l2 * weights[index];
            weights[index] -= learning_rate * gradient;
        }
        bias -= learning_rate * bias_gradient * scale;
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    Activation activation;
    vector<double> input_vector;
    vector<double> output_vector;   // post-dropout values handed downstream
    vector<double> dropout_mask;    // empty when dropout is inactive

    Layer() : activation(Activation::Linear) {}

    Layer(int neuron_count, int input_size, Activation activation_in, mt19937& rng)
        : activation(activation_in) {
        neurons.reserve(neuron_count);
        for (int index = 0; index < neuron_count; ++index) {
            neurons.emplace_back(input_size, rng);
        }
    }

    Layer(int neuron_count, int input_size, const string& activation_in, mt19937& rng)
        : Layer(neuron_count, input_size, parse_activation(activation_in), rng) {}

    vector<double> forward(const vector<double>& input) {
        return forward(input, false, 0.0, nullptr);
    }
/*
-----------------------------------------------------------------------------
                            DATA FLOW: FORWARD
-----------------------------------------------------------------------------

    forward_pass() carries one vector through the network, reassigning it at
    each layer, so layer i's output vector IS layer i+1's input vector:

        input :x:
          -> hidden[0]  : :z: = W:x: + :b:  ->  :a: = f(:z:)
          -> hidden[1]  : :z: = W:a: + :b:  ->  :a: = f(:z:)
          -> ...
          -> output     : :z: = W:a: + :b:  ->  :p: = f(:z:)

    Each neuron stores the input it saw in input_cache; the backward pass
    needs it, and it is gone once the next forward pass runs.
*/
    vector<double> forward(const vector<double>& input, bool training, double dropout_rate, mt19937* rng) {
        input_vector = input;
        output_vector.assign(neurons.size(), 0.0);

        for (Neuron& neuron : neurons) {
            neuron.compute_pre_activation(input);
        }

        if (activation == Activation::Softmax) {
            // Shift by the max before exp so large logits cannot overflow.
            double max_pre = -numeric_limits<double>::infinity();
            for (const Neuron& neuron : neurons) {
                max_pre = max(max_pre, neuron.pre_activation);
            }
            double sum = 0.0;
            for (size_t index = 0; index < neurons.size(); ++index) {
                double value = exp(neurons[index].pre_activation - max_pre);
                output_vector[index] = value;
                sum += value;
            }
            for (size_t index = 0; index < neurons.size(); ++index) {
                output_vector[index] /= sum;
                neurons[index].output = output_vector[index];
            }
        } else {
            for (size_t index = 0; index < neurons.size(); ++index) {
                neurons[index].output = Neuron::activate(neurons[index].pre_activation, activation);
                output_vector[index] = neurons[index].output;
            }
        }

        // Inverted dropout: scale the survivors so inference needs no rescaling.
        if (training && dropout_rate > 0.0 && rng != nullptr) {
            if (dropout_rate >= 1.0) {
                throw invalid_argument("Dropout rate must be < 1.0.");
            }
            dropout_mask.assign(neurons.size(), 0.0);
            bernoulli_distribution keep(1.0 - dropout_rate);
            double scale = 1.0 / (1.0 - dropout_rate);
            for (size_t index = 0; index < neurons.size(); ++index) {
                double mask = keep(*rng) ? scale : 0.0;
                dropout_mask[index] = mask;
                output_vector[index] *= mask;
            }
        } else {
            dropout_mask.clear();
        }

        return output_vector;
    }
/*
-----------------------------------------------------------------------------
                            DATA FLOW: BACKWARD
-----------------------------------------------------------------------------

    Gradients flow the other way. At each layer, three things happen:

        1. local gradient   dL/d:z:_i = dL/d:a:_i * f'(:z:_i)
        2. weight gradient  dL/dw_ij  = dL/d:z:_i * input_cache_j
                            dL/db_i   = dL/d:z:_i          (its input is 1)
        3. pass back        dL/d:a:_j = sum_i(dL/d:z:_i * w_ij)

    Step 3 is the transpose of the weight matrix, and it must read the
    weights BEFORE they are updated - see INVARIANTS.

    Softmax + Categorical Cross Entropy skip step 1 entirely. Substituting
    the softmax Jacobian into d_CCE, the p_i terms cancel and the whole
    chain collapses to:

        dL/d:z: = :p: - :y:

    That is why Layer::backward_local() exists: it takes gradients already
    with respect to :z:, letting the fused path inject :p: - :y: directly.
    The constructor rejects softmax without CCE, CCE without softmax, and
    softmax as a hidden activation, because none of those have a correct
    path through this code.

    */
    // Accumulates weight/bias gradients and returns dL/d(input).
    // local_gradients are gradients with respect to each neuron's pre-activation.

    vector<double> backward_local(const vector<double>& local_gradients) {
        if (local_gradients.size() != neurons.size()) {
            throw invalid_argument("Gradient size must match neuron count.");
        }

        vector<double> gradient_wrt_input(input_vector.size(), 0.0);

        for (size_t neuron_index = 0; neuron_index < neurons.size(); ++neuron_index) {
            Neuron& neuron = neurons[neuron_index];
            double local_gradient = local_gradients[neuron_index];

            if (neuron.weight_gradients.size() != neuron.weights.size()) {
                neuron.weight_gradients.assign(neuron.weights.size(), 0.0);
            }
            for (size_t weight_index = 0; weight_index < neuron.weights.size(); ++weight_index) {
                neuron.weight_gradients[weight_index] += local_gradient * neuron.input_cache[weight_index];
                gradient_wrt_input[weight_index] += local_gradient * neuron.weights[weight_index];
            }
            neuron.bias_gradient += local_gradient;
        }

        return gradient_wrt_input;
    }

    vector<double> backward(const vector<double>& gradient_wrt_output) {
        if (gradient_wrt_output.size() != neurons.size()) {
            throw invalid_argument("Gradient size must match neuron count.");
        }
        if (activation == Activation::Softmax) {
            throw invalid_argument("Softmax layers must be back-propagated through the fused loss path.");
        }

        vector<double> local_gradients(neurons.size(), 0.0);
        for (size_t index = 0; index < neurons.size(); ++index) {
            double upstream = gradient_wrt_output[index];
            if (!dropout_mask.empty()) {
                upstream *= dropout_mask[index];   // dropped units pass no gradient
            }
            local_gradients[index] = upstream * Neuron::activation_derivative(
                neurons[index].pre_activation, neurons[index].output, activation);
        }
        return backward_local(local_gradients);
    }

    // Legacy immediate-update form: accumulate this sample, then apply at once.
    vector<double> backward(const vector<double>& gradient_wrt_output, double learning_rate) {
        zero_gradients();
        vector<double> gradient_wrt_input = backward(gradient_wrt_output);
        apply_gradients(learning_rate, 1.0, 0.0);
        return gradient_wrt_input;
    }

    void zero_gradients() {
        for (Neuron& neuron : neurons) {
            neuron.zero_gradients();
        }
    }

    void apply_gradients(double learning_rate, double scale, double l2) {
        for (Neuron& neuron : neurons) {
            neuron.apply_gradients(learning_rate, scale, l2);
        }
    }
};
/*
-----------------------------------------------------------------------------
                                TRAINING
-----------------------------------------------------------------------------

    TrainOptions          default    meaning
      epochs              100        passes over the training data
      learning_rate       0.1        step size
      batch_size          1          1 reproduces pure per-sample SGD
      shuffle             true       reshuffle order every epoch
      l2                  0.0        weight decay (never applied to biases)
      dropout             0.0        hidden layers only, training only
      validation_split    0.0        fraction held out, taken after a shuffle
      log_interval        0          0 picks a readable interval
      verbose             true       print per-epoch progress

    TrainHistory returns train_loss and validation_loss per epoch, plus
    best_validation_loss / best_epoch. Nothing acts on those - there is no
    early stopping - so a caller watching for overfitting must read them.

    Dropout is inverted: survivors are scaled by 1/(1-p) during training, so
    inference needs no rescaling and predict() is always deterministic.

    The original five-argument train(inputs, targets, epochs, learning_rate,
    log_interval) still exists and is unchanged: one sample per update, in
    fixed order, no regularization.

    Weights use He initialization, sqrt(2/fan_in); biases start at zero.
    Inputs are assumed to be roughly unit-scale - unnormalized features
    train noticeably worse.
*/

struct TrainOptions {
    int epochs = 100;
    double learning_rate = 0.1;
    int batch_size = 1;              // 1 reproduces pure per-sample SGD
    bool shuffle = true;
    double l2 = 0.0;                 // weight decay coefficient
    double dropout = 0.0;            // applied to hidden layers, training only
    double validation_split = 0.0;   // fraction held out for validation
    int log_interval = 0;            // 0 picks a readable interval automatically
    bool verbose = true;
};

struct TrainHistory {
    vector<double> train_loss;
    vector<double> validation_loss;
    double best_validation_loss = numeric_limits<double>::infinity();
    int best_epoch = 0;
    int training_samples = 0;
    int validation_samples = 0;
};

class NeuralNetwork {
public:
    vector<Layer> hidden_layers;
    Layer output_layer;
    Loss loss;
    uint64_t seed;              // the seed actually used, so any run is reproducible
    mt19937 rng;

    NeuralNetwork(
        int input_size,
        int hidden_layer_count,
        int layer_width,
        int output_layer_width,
        Activation hidden_layer_activation,
        Activation output_activation,
        Loss loss_in,
        int64_t requested_seed = -1
    ) : loss(loss_in) {
        if (input_size <= 0 || hidden_layer_count < 0 || layer_width <= 0 || output_layer_width <= 0) {
            throw invalid_argument("Invalid network dimensions.");
        }
        if (hidden_layer_activation == Activation::Softmax) {
            throw invalid_argument("Softmax is only supported as an output activation.");
        }
        if (output_activation == Activation::Softmax && loss != Loss::CategoricalCrossEntropy) {
            throw invalid_argument("Softmax output requires categorical_cross_entropy.");
        }
        if (loss == Loss::CategoricalCrossEntropy && output_activation != Activation::Softmax) {
            throw invalid_argument("categorical_cross_entropy requires a softmax output.");
        }

        if (requested_seed < 0) {
            random_device random_device_source;
            seed = (static_cast<uint64_t>(random_device_source()) << 32) ^ random_device_source();
        } else {
            seed = static_cast<uint64_t>(requested_seed);
        }
        rng.seed(static_cast<mt19937::result_type>(seed));

        int current_input_size = input_size;
        hidden_layers.reserve(hidden_layer_count);
        for (int layer_index = 0; layer_index < hidden_layer_count; ++layer_index) {
            hidden_layers.emplace_back(layer_width, current_input_size, hidden_layer_activation, rng);
            current_input_size = layer_width;
        }
        output_layer = Layer(output_layer_width, current_input_size, output_activation, rng);
    }

    NeuralNetwork(
        int input_size,
        int hidden_layer_count,
        int layer_width,
        int output_layer_width,
        const string& hidden_layer_activation,
        const string& output_activation,
        const string& loss_in,
        int64_t requested_seed = -1
    ) : NeuralNetwork(input_size, hidden_layer_count, layer_width, output_layer_width,
                      parse_activation(hidden_layer_activation), parse_activation(output_activation),
                      parse_loss(loss_in), requested_seed) {}

    // Used by load(); builds an empty shell to be populated from a file.
    NeuralNetwork() : loss(Loss::MeanSquaredError), seed(0) { rng.seed(0); }

    int input_size() const {
        const Layer& first = hidden_layers.empty() ? output_layer : hidden_layers.front();
        if (first.neurons.empty()) {
            return 0;
        }
        return static_cast<int>(first.neurons.front().weights.size());
    }

    int output_size() const { return static_cast<int>(output_layer.neurons.size()); }

    vector<double> forward_pass(const vector<double>& input_vector, bool training = false, double dropout = 0.0) {
        vector<double> activations = input_vector;
        for (Layer& layer : hidden_layers) {
            activations = layer.forward(activations, training, dropout, &rng);
        }
        return output_layer.forward(activations, false, 0.0, nullptr);
    }

    double calculate_loss(const vector<double>& predictions, const vector<double>& target_vector) const {
        if (predictions.size() != target_vector.size()) {
            throw invalid_argument("Prediction and target sizes must match.");
        }

        const double epsilon = 1e-9;
        double total = 0.0;

        switch (loss) {
            case Loss::MeanSquaredError:
                for (size_t index = 0; index < target_vector.size(); ++index) {
                    double difference = predictions[index] - target_vector[index];
                    total += difference * difference;
                }
                return total / static_cast<double>(target_vector.size());

            case Loss::BinaryCrossEntropy:
                for (size_t index = 0; index < target_vector.size(); ++index) {
                    double prediction = clamp(predictions[index], epsilon, 1.0 - epsilon);
                    total += -(target_vector[index] * log(prediction) +
                               (1.0 - target_vector[index]) * log(1.0 - prediction));
                }
                return total / static_cast<double>(target_vector.size());

            case Loss::CategoricalCrossEntropy:
                // Summed over classes, the standard definition for a one-hot target.
                for (size_t index = 0; index < target_vector.size(); ++index) {
                    double prediction = clamp(predictions[index], epsilon, 1.0);
                    total += -target_vector[index] * log(prediction);
                }
                return total;
        }
        throw invalid_argument("Unknown loss enum value.");
    }

    vector<double> loss_gradient(const vector<double>& predictions, const vector<double>& target_vector) const {
        if (predictions.size() != target_vector.size()) {
            throw invalid_argument("Prediction and target sizes must match.");
        }

        vector<double> gradient(predictions.size(), 0.0);
        const double epsilon = 1e-9;
        const double output_count = static_cast<double>(predictions.size());

        switch (loss) {
            case Loss::MeanSquaredError:
                for (size_t index = 0; index < predictions.size(); ++index) {
                    gradient[index] = 2.0 * (predictions[index] - target_vector[index]) / output_count;
                }
                return gradient;

            case Loss::BinaryCrossEntropy:
                for (size_t index = 0; index < predictions.size(); ++index) {
                    double prediction = clamp(predictions[index], epsilon, 1.0 - epsilon);
                    gradient[index] = (prediction - target_vector[index]) /
                                      (prediction * (1.0 - prediction) * output_count);
                }
                return gradient;

            case Loss::CategoricalCrossEntropy:
                // Softmax + CCE collapse to (p - t) at the pre-activation, which is
                // what accumulate_gradients uses; this branch is the raw dL/dp.
                for (size_t index = 0; index < predictions.size(); ++index) {
                    double prediction = clamp(predictions[index], epsilon, 1.0);
                    gradient[index] = -target_vector[index] / prediction;
                }
                return gradient;
        }
        throw invalid_argument("Unknown loss enum value.");
    }

    // Adds one sample's gradients to the accumulators without touching weights.
    void accumulate_gradients(const vector<double>& target_vector) {
        vector<double> gradient;

        if (output_layer.activation == Activation::Softmax && loss == Loss::CategoricalCrossEntropy) {
            // Fused softmax + categorical cross-entropy: dL/dz = p - t.
            const vector<double>& predictions = output_layer.output_vector;
            if (predictions.size() != target_vector.size()) {
                throw invalid_argument("Prediction and target sizes must match.");
            }
            vector<double> local(predictions.size(), 0.0);
            for (size_t index = 0; index < predictions.size(); ++index) {
                local[index] = predictions[index] - target_vector[index];
            }
            gradient = output_layer.backward_local(local);
        } else {
            gradient = loss_gradient(output_layer.output_vector, target_vector);
            gradient = output_layer.backward(gradient);
        }

        for (int layer_index = static_cast<int>(hidden_layers.size()) - 1; layer_index >= 0; --layer_index) {
            gradient = hidden_layers[layer_index].backward(gradient);
        }
    }

    void zero_gradients() {
        for (Layer& layer : hidden_layers) {
            layer.zero_gradients();
        }
        output_layer.zero_gradients();
    }

    void apply_gradients(double learning_rate, double scale, double l2) {
        for (Layer& layer : hidden_layers) {
            layer.apply_gradients(learning_rate, scale, l2);
        }
        output_layer.apply_gradients(learning_rate, scale, l2);
    }

    // Single-sample update, unchanged in behaviour from the original engine.
    void backward_pass(const vector<double>& target_vector, double learning_rate) {
        zero_gradients();
        accumulate_gradients(target_vector);
        apply_gradients(learning_rate, 1.0, 0.0);
    }

    double evaluate(const vector<vector<double>>& input_data,
                    const vector<vector<double>>& target_data) {
        if (input_data.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (size_t index = 0; index < input_data.size(); ++index) {
            total += calculate_loss(forward_pass(input_data[index]), target_data[index]);
        }
        return total / static_cast<double>(input_data.size());
    }

    double accuracy(const vector<vector<double>>& input_data,
                    const vector<vector<double>>& target_data) {
        if (input_data.empty()) {
            return 0.0;
        }
        int correct = 0;
        for (size_t index = 0; index < input_data.size(); ++index) {
            vector<double> prediction = forward_pass(input_data[index]);
            size_t predicted = static_cast<size_t>(
                max_element(prediction.begin(), prediction.end()) - prediction.begin());
            size_t actual = static_cast<size_t>(
                max_element(target_data[index].begin(), target_data[index].end()) - target_data[index].begin());
            if (predicted == actual) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(input_data.size());
    }

    TrainHistory train(
        const vector<vector<double>>& input_data,
        const vector<vector<double>>& target_data,
        const TrainOptions& options
    ) {
        if (input_data.empty() || input_data.size() != target_data.size()) {
            throw invalid_argument("Input and target datasets must be non-empty and same size.");
        }
        if (options.epochs <= 0 || options.learning_rate <= 0.0) {
            throw invalid_argument("Epochs and learning rate must be positive.");
        }
        if (options.batch_size <= 0) {
            throw invalid_argument("Batch size must be positive.");
        }
        if (options.l2 < 0.0) {
            throw invalid_argument("L2 coefficient must be non-negative.");
        }
        if (options.dropout < 0.0 || options.dropout >= 1.0) {
            throw invalid_argument("Dropout rate must be in [0, 1).");
        }
        if (options.validation_split < 0.0 || options.validation_split >= 1.0) {
            throw invalid_argument("Validation split must be in [0, 1).");
        }

        vector<size_t> indices(input_data.size());
        iota(indices.begin(), indices.end(), 0);

        // Hold out the validation slice from a shuffled order so the split is not
        // an artefact of how the caller happened to order the data.
        size_t validation_count = static_cast<size_t>(
            static_cast<double>(input_data.size()) * options.validation_split);
        if (options.validation_split > 0.0 && validation_count == 0) {
            validation_count = 1;
        }
        if (validation_count >= input_data.size()) {
            throw invalid_argument("Validation split leaves no training data.");
        }
        if (validation_count > 0) {
            shuffle(indices.begin(), indices.end(), rng);
        }

        vector<size_t> validation_indices(indices.begin(), indices.begin() + validation_count);
        vector<size_t> training_indices(indices.begin() + validation_count, indices.end());

        vector<vector<double>> validation_inputs, validation_targets;
        for (size_t index : validation_indices) {
            validation_inputs.push_back(input_data[index]);
            validation_targets.push_back(target_data[index]);
        }

        TrainHistory history;
        history.training_samples = static_cast<int>(training_indices.size());
        history.validation_samples = static_cast<int>(validation_indices.size());

        int log_interval = options.log_interval > 0
            ? options.log_interval
            : max(1, options.epochs / 10);

        for (int epoch = 1; epoch <= options.epochs; ++epoch) {
            if (options.shuffle) {
                shuffle(training_indices.begin(), training_indices.end(), rng);
            }

            double epoch_loss = 0.0;
            size_t position = 0;
            while (position < training_indices.size()) {
                size_t batch_end = min(position + static_cast<size_t>(options.batch_size),
                                       training_indices.size());
                zero_gradients();

                for (size_t offset = position; offset < batch_end; ++offset) {
                    size_t sample = training_indices[offset];
                    vector<double> prediction = forward_pass(input_data[sample], true, options.dropout);
                    epoch_loss += calculate_loss(prediction, target_data[sample]);
                    accumulate_gradients(target_data[sample]);
                }

                double batch_count = static_cast<double>(batch_end - position);
                apply_gradients(options.learning_rate, 1.0 / batch_count, options.l2);
                position = batch_end;
            }

            epoch_loss /= static_cast<double>(training_indices.size());
            history.train_loss.push_back(epoch_loss);

            double validation_loss = numeric_limits<double>::quiet_NaN();
            if (!validation_inputs.empty()) {
                validation_loss = evaluate(validation_inputs, validation_targets);
                history.validation_loss.push_back(validation_loss);
                if (validation_loss < history.best_validation_loss) {
                    history.best_validation_loss = validation_loss;
                    history.best_epoch = epoch;
                }
            }

            if (options.verbose && (epoch == 1 || epoch % log_interval == 0 || epoch == options.epochs)) {
                cout << "Epoch " << epoch << " - Loss: " << fixed << setprecision(6) << epoch_loss;
                if (!validation_inputs.empty()) {
                    cout << " - Val loss: " << validation_loss;
                }
                cout << endl;
            }
        }

        return history;
    }

    // Original signature: fixed order, one sample per update, no regularization.
    void train(
        const vector<vector<double>>& input_data,
        const vector<vector<double>>& target_data,
        int epochs,
        double learning_rate,
        int log_interval = 200
    ) {
        TrainOptions options;
        options.epochs = epochs;
        options.learning_rate = learning_rate;
        options.batch_size = 1;
        options.shuffle = false;
        options.log_interval = log_interval;
        options.verbose = true;
        train(input_data, target_data, options);
    }

    vector<double> predict(const vector<double>& input) {
        return forward_pass(input, false, 0.0);
    }

    // ---- persistence -------------------------------------------------------
    // 17 significant digits round-trips an IEEE-754 double exactly, so a
    // reloaded network predicts bit-for-bit identically.
/*
-----------------------------------------------------------------------------
                              MODEL FILE FORMAT
-----------------------------------------------------------------------------

    Plain text, versioned. Values carry 17 significant digits, which is
    enough to round-trip an IEEE-754 double exactly, so a reloaded network
    predicts bit-for-bit identically to the one that was saved.

        GRADIEX-MODEL 1
        loss categorical_cross_entropy
        seed 20260821
        layers 3
        layer 0 24 2 tanh          <- index, neurons, fan-in, activation
        <bias> <w0> <w1> ...       <- one line per neuron
        ...
    Bump the version if this layout changes; load() rejects versions it does
    not recognize, along with truncated, malformed, and missing files.
*/

    void save(const string& path) const {
        ofstream file(path);
        if (!file) {
            throw runtime_error("Could not open file for writing: " + path);
        }
        file << setprecision(17);

        file << "GRADIEX-MODEL 1\n";
        file << "loss " << loss_name(loss) << "\n";
        file << "seed " << seed << "\n";
        file << "layers " << (hidden_layers.size() + 1) << "\n";

        vector<const Layer*> all_layers;
        for (const Layer& layer : hidden_layers) {
            all_layers.push_back(&layer);
        }
        all_layers.push_back(&output_layer);

        for (size_t layer_index = 0; layer_index < all_layers.size(); ++layer_index) {
            const Layer& layer = *all_layers[layer_index];
            size_t fan_in = layer.neurons.empty() ? 0 : layer.neurons.front().weights.size();
            file << "layer " << layer_index << " " << layer.neurons.size() << " " << fan_in
                 << " " << activation_name(layer.activation) << "\n";
            for (const Neuron& neuron : layer.neurons) {
                file << neuron.bias;
                for (double weight : neuron.weights) {
                    file << " " << weight;
                }
                file << "\n";
            }
        }

        if (!file) {
            throw runtime_error("Failed while writing model to: " + path);
        }
    }

    static NeuralNetwork load(const string& path) {
        ifstream file(path);
        if (!file) {
            throw runtime_error("Could not open file for reading: " + path);
        }

        string token;
        int version = 0;
        if (!(file >> token >> version) || token != "GRADIEX-MODEL") {
            throw runtime_error("Not a GradieX model file: " + path);
        }
        if (version != 1) {
            throw runtime_error("Unsupported model version: " + to_string(version));
        }

        NeuralNetwork network;
        string loss_text;
        uint64_t stored_seed = 0;
        size_t layer_count = 0;

        if (!(file >> token >> loss_text) || token != "loss") {
            throw runtime_error("Malformed model file: expected loss.");
        }
        network.loss = parse_loss(loss_text);

        if (!(file >> token >> stored_seed) || token != "seed") {
            throw runtime_error("Malformed model file: expected seed.");
        }
        network.seed = stored_seed;
        network.rng.seed(static_cast<mt19937::result_type>(stored_seed));

        if (!(file >> token >> layer_count) || token != "layers" || layer_count == 0) {
            throw runtime_error("Malformed model file: expected layers.");
        }

        vector<Layer> parsed;
        for (size_t expected = 0; expected < layer_count; ++expected) {
            size_t declared_index = 0, neuron_count = 0, fan_in = 0;
            string activation_text;
            if (!(file >> token >> declared_index >> neuron_count >> fan_in >> activation_text) ||
                token != "layer") {
                throw runtime_error("Malformed model file: expected layer header.");
            }
            if (declared_index != expected) {
                throw runtime_error("Malformed model file: layers out of order.");
            }

            Layer layer;
            layer.activation = parse_activation(activation_text);
            layer.neurons.resize(neuron_count);
            for (size_t neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
                Neuron& neuron = layer.neurons[neuron_index];
                if (!(file >> neuron.bias)) {
                    throw runtime_error("Malformed model file: truncated bias.");
                }
                neuron.weights.resize(fan_in);
                for (size_t weight_index = 0; weight_index < fan_in; ++weight_index) {
                    if (!(file >> neuron.weights[weight_index])) {
                        throw runtime_error("Malformed model file: truncated weights.");
                    }
                }
                neuron.weight_gradients.assign(fan_in, 0.0);
            }
            parsed.push_back(std::move(layer));
        }

        network.output_layer = parsed.back();
        parsed.pop_back();
        network.hidden_layers = parsed;
        return network;
    }
};

// ---------------------------------------------------------------------------

/*

-----------------------------------------------------------------------------
  QUICK START
-----------------------------------------------------------------------------

    NeuralNetwork net(n_features, 2, 32, n_classes,
                      "ReLU", "softmax", "categorical_cross_entropy", 42);

    TrainOptions o;
    o.epochs = 300;  o.learning_rate = 0.05;  o.batch_size = 32;
    o.l2 = 1e-4;     o.validation_split = 0.2;

    TrainHistory h = net.train(X, Y, o);
    net.save("model.gx");
    NeuralNetwork back = NeuralNetwork::load("model.gx");   // predicts the same

    Pass a seed to fix initialization, shuffling, and dropout masks; omit it
    (or pass -1) to seed from random_device. The seed in use is readable as
    net.seed and is written into saved models.

-----------------------------------------------------------------------------
  INVARIANTS - changing these silently breaks correctness
-----------------------------------------------------------------------------

    1. Gradients accumulate; no weight moves until apply_gradients(). This is
       what makes mini-batching exact, and it is what guarantees step 3 of
       the backward pass reads pre-update weights. Updating a layer's weights
       before propagating through it corrupts every layer behind it.

    2. input_cache must hold the input from THIS forward pass. Weight
       gradients are meaningless against a stale cache.

    3. Neuron::output is the activation BEFORE the dropout mask;
       Layer::output_vector is the value AFTER it. Sigmoid and tanh
       derivatives are written in terms of the unmasked output, so masking
       Neuron::output would corrupt them. With dropout off the two are equal.

    4. Softmax is layer-wide and fused with CCE. Neuron::activate and
       Neuron::activation_derivative throw rather than return a wrong answer.

    5. Activation and Loss are enums on the hot path. Strings are parsed once
       at the API boundary; do not push a string comparison back into
       Layer::forward or the per-neuron loops.

    6. The five-argument train() is bit-identical to a manual fixed-order,
       one-sample-at-a-time loop. A test asserts this.

-----------------------------------------------------------------------------
  NOT SUPPORTED
-----------------------------------------------------------------------------

    Optimizers other than plain SGD (no momentum, Adam, or LR schedule),
    per-layer widths or dropout rates from the constructor, early stopping,
    convolution, recurrence, attention, embeddings, skip connections, batch
    or layer normalization, sparse input, streaming datasets, GPU, threading,
    or SIMD. The inner loops are scalar double arithmetic.

    Mean Absolute Error is not implemented. Earlier revisions of this header
    listed its derivative; it was never in the Loss enum.

*/
static void print_rule(const string& title) {
    cout << "\n" << string(62, '=') << "\n" << title << "\n" << string(62, '=') << endl;
}

int main() {
    print_rule("XOR - binary classification, per-sample SGD (seeded)");

    vector<vector<double>> xor_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<vector<double>> xor_targets = {{0.0}, {1.0}, {1.0}, {0.0}};

    NeuralNetwork network(2, 2, 4, 1, "tanh", "sigmoid", "binary_cross_entropy", 42);
    network.train(xor_inputs, xor_targets, 5000, 0.1, 500);

    cout << "\nPredictions after training:" << endl;
    for (const vector<double>& sample : xor_inputs) {
        vector<double> prediction = network.predict(sample);
        cout << sample[0] << " XOR " << sample[1] << " -> " << fixed << setprecision(6)
             << prediction[0] << endl;
    }

    // ---- multi-class demo: mini-batches, shuffling, L2, dropout, val split ----
    print_rule("Three-class spirals - softmax + categorical cross-entropy");

    const double pi = acos(-1.0);
    const int classes = 3;
    const int per_class = 120;
    vector<vector<double>> spiral_inputs;
    vector<vector<double>> spiral_targets;
    {
        mt19937 data_rng(7);
        normal_distribution<double> noise(0.0, 0.055);
        for (int label = 0; label < classes; ++label) {
            for (int index = 0; index < per_class; ++index) {
                double fraction = static_cast<double>(index) / static_cast<double>(per_class);
                double radius = 0.15 + 0.85 * fraction;
                double angle = fraction * 2.4 + label * (2.0 * pi / classes);
                double x = radius * cos(angle) + noise(data_rng);
                double y = radius * sin(angle) + noise(data_rng);
                spiral_inputs.push_back({x, y});
                vector<double> one_hot(classes, 0.0);
                one_hot[label] = 1.0;
                spiral_targets.push_back(one_hot);
            }
        }
    }
    cout << "Dataset: " << spiral_inputs.size() << " points, " << classes
         << " interleaved spiral arms, 2 features" << endl;

    NeuralNetwork classifier(2, 2, 24, classes, "tanh", "softmax",
                             "categorical_cross_entropy", 20260821);

    TrainOptions options;
    options.epochs = 400;
    options.learning_rate = 0.05;
    options.batch_size = 16;
    options.shuffle = true;
    options.l2 = 1e-5;
    options.dropout = 0.0;           // a 24-wide net on 360 points does not need it
    options.validation_split = 0.25;
    options.log_interval = 50;

    cout << "Config: batch_size=" << options.batch_size << ", shuffle=on, L2="
         << scientific << setprecision(0) << options.l2
         << fixed << setprecision(2)
         << ", validation_split=" << options.validation_split
         << ", seed=" << classifier.seed << "\n" << endl;

    TrainHistory history = classifier.train(spiral_inputs, spiral_targets, options);

    cout << "\nTrain samples: " << history.training_samples
         << "   Validation samples: " << history.validation_samples << endl;
    cout << "First-epoch loss: " << fixed << setprecision(6) << history.train_loss.front()
         << "   Final: " << history.train_loss.back() << endl;
    cout << "Best validation loss: " << history.best_validation_loss
         << " at epoch " << history.best_epoch << endl;
    cout << "Accuracy - overall: " << setprecision(2)
         << 100.0 * classifier.accuracy(spiral_inputs, spiral_targets) << "%" << endl;

    // ---- persistence round-trip -------------------------------------------
    print_rule("Save / load round-trip");

    const string model_path = "spiral_model.gx";
    classifier.save(model_path);
    cout << "Saved trained model to " << model_path << endl;

    NeuralNetwork reloaded = NeuralNetwork::load(model_path);
    cout << "Reloaded: " << reloaded.hidden_layers.size() << " hidden layers, "
         << reloaded.output_size() << " outputs, loss=" << loss_name(reloaded.loss) << endl;

    bool identical = true;
    double worst_difference = 0.0;
    for (size_t index = 0; index < spiral_inputs.size(); ++index) {
        vector<double> before = classifier.predict(spiral_inputs[index]);
        vector<double> after = reloaded.predict(spiral_inputs[index]);
        for (size_t output_index = 0; output_index < before.size(); ++output_index) {
            double difference = fabs(before[output_index] - after[output_index]);
            worst_difference = max(worst_difference, difference);
            if (before[output_index] != after[output_index]) {
                identical = false;
            }
        }
    }
    cout << "Predictions across all " << spiral_inputs.size() << " points: "
         << (identical ? "bit-for-bit identical" : "DIFFER") << " (max delta "
         << scientific << setprecision(1) << worst_difference << ")" << endl;
    cout << "Reloaded accuracy: " << fixed << setprecision(2)
         << 100.0 * reloaded.accuracy(spiral_inputs, spiral_targets) << "%" << endl;

    cout << "\nSample predictions from the reloaded model:" << endl;
    cout << "      x         y     ->   p(c0)   p(c1)   p(c2)   pred  true" << endl;
    for (int label = 0; label < classes; ++label) {
        size_t index = static_cast<size_t>(label * per_class + per_class / 2);
        vector<double> probabilities = reloaded.predict(spiral_inputs[index]);
        size_t predicted = static_cast<size_t>(
            max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
        cout << setw(9) << fixed << setprecision(4) << spiral_inputs[index][0]
             << setw(10) << spiral_inputs[index][1] << "    ";
        double sum = 0.0;
        for (double probability : probabilities) {
            cout << setw(8) << setprecision(4) << probability;
            sum += probability;
        }
        cout << setw(7) << predicted << setw(6) << label
             << "   (sums to " << setprecision(6) << sum << ")" << endl;
    }

    return 0;
}
