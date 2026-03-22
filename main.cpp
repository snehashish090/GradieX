#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class Neuron {
public:
    vector<double> weights;
    double bias;
    vector<double> input_cache;
    double pre_activation;
    double output;

    Neuron() : bias(0.0), pre_activation(0.0), output(0.0) {}

    Neuron(int input_size, mt19937& rng) : bias(0.0), pre_activation(0.0), output(0.0) {
        normal_distribution<double> dist(0.0, sqrt(2.0 / static_cast<double>(max(1, input_size))));
        weights.resize(input_size);
        for (double& weight : weights) {
            weight = dist(rng);
        }
    }

    static double activate(double value, const string& activation) {
        if (activation == "sigmoid") {
            if (value >= 0.0) {
                return 1.0 / (1.0 + exp(-value));
            }
            double exp_value = exp(value);
            return exp_value / (1.0 + exp_value);
        }
        if (activation == "tanh") {
            return tanh(value);
        }
        if (activation == "ReLU") {
            return max(0.0, value);
        }
        if (activation == "linear") {
            return value;
        }
        if (activation == "leaky_ReLU") {
            return value > 0.0 ? value : 0.01 * value;
        }
        throw invalid_argument("Unsupported activation function: " + activation);
    }

    static double activation_derivative(double pre_activation_value, double output_value, const string& activation) {
        if (activation == "sigmoid") {
            return output_value * (1.0 - output_value);
        }
        if (activation == "tanh") {
            return 1.0 - (output_value * output_value);
        }
        if (activation == "ReLU") {
            return pre_activation_value > 0.0 ? 1.0 : 0.0;
        }
        if (activation == "linear") {
            return 1.0;
        }
        if (activation == "leaky_ReLU") {
            return pre_activation_value > 0.0 ? 1.0 : 0.01;
        }
        throw invalid_argument("Unsupported activation function: " + activation);
    }

    double forward(const vector<double>& input, const string& activation) {
        if (input.size() != weights.size()) {
            throw invalid_argument("Input size does not match neuron weight size.");
        }

        input_cache = input;
        pre_activation = bias;
        for (size_t index = 0; index < input.size(); ++index) {
            pre_activation += weights[index] * input[index];
        }

        output = activate(pre_activation, activation);
        return output;
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    string activation_function;
    vector<double> input_vector;
    vector<double> output_vector;

    Layer() {}

    Layer(int neuron_count, int input_size, const string& activation, mt19937& rng)
        : activation_function(activation) {
        neurons.reserve(neuron_count);
        for (int index = 0; index < neuron_count; ++index) {
            neurons.emplace_back(input_size, rng);
        }
    }

    vector<double> forward(const vector<double>& input) {
        input_vector = input;
        output_vector.clear();
        output_vector.reserve(neurons.size());

        for (Neuron& neuron : neurons) {
            output_vector.push_back(neuron.forward(input, activation_function));
        }

        return output_vector;
    }

    vector<double> backward(const vector<double>& gradient_wrt_output, double learning_rate) {
        if (gradient_wrt_output.size() != neurons.size()) {
            throw invalid_argument("Gradient size must match neuron count.");
        }

        vector<double> gradient_wrt_input(input_vector.size(), 0.0);

        for (size_t neuron_index = 0; neuron_index < neurons.size(); ++neuron_index) {
            Neuron& neuron = neurons[neuron_index];
            double local_gradient = gradient_wrt_output[neuron_index] *
                Neuron::activation_derivative(neuron.pre_activation, neuron.output, activation_function);

            for (size_t weight_index = 0; weight_index < neuron.weights.size(); ++weight_index) {
                double old_weight = neuron.weights[weight_index];
                double gradient_wrt_weight = local_gradient * neuron.input_cache[weight_index];
                neuron.weights[weight_index] -= learning_rate * gradient_wrt_weight;
                gradient_wrt_input[weight_index] += local_gradient * old_weight;
            }

            neuron.bias -= learning_rate * local_gradient;
        }

        return gradient_wrt_input;
    }
};

class NeuralNetwork {
public:
    vector<Layer> hidden_layers;
    Layer output_layer;
    string loss_function;

    NeuralNetwork(
        int input_size,
        int hidden_layer_count,
        int layer_width,
        int output_layer_width,
        const string& hidden_layer_activation,
        const string& output_activation,
        const string& loss_function
    ) : loss_function(loss_function) {
        if (input_size <= 0 || hidden_layer_count < 0 || layer_width <= 0 || output_layer_width <= 0) {
            throw invalid_argument("Invalid network dimensions.");
        }

        random_device random_device_source;
        mt19937 rng(random_device_source());

        int current_input_size = input_size;
        hidden_layers.reserve(hidden_layer_count);

        for (int layer_index = 0; layer_index < hidden_layer_count; ++layer_index) {
            hidden_layers.emplace_back(layer_width, current_input_size, hidden_layer_activation, rng);
            current_input_size = layer_width;
        }

        output_layer = Layer(output_layer_width, current_input_size, output_activation, rng);
    }

    vector<double> forward_pass(const vector<double>& input_vector) {
        vector<double> activations = input_vector;

        for (Layer& layer : hidden_layers) {
            activations = layer.forward(activations);
        }

        return output_layer.forward(activations);
    }

    double calculate_loss(const vector<double>& predictions, const vector<double>& target_vector) const {
        if (predictions.size() != target_vector.size()) {
            throw invalid_argument("Prediction and target sizes must match.");
        }

        const double epsilon = 1e-9;
        double loss = 0.0;

        if (loss_function == "mean_squared_error") {
            for (size_t index = 0; index < target_vector.size(); ++index) {
                double difference = predictions[index] - target_vector[index];
                loss += difference * difference;
            }
            return loss / static_cast<double>(target_vector.size());
        }

        if (loss_function == "binary_cross_entropy") {
            for (size_t index = 0; index < target_vector.size(); ++index) {
                double prediction = clamp(predictions[index], epsilon, 1.0 - epsilon);
                loss += -(target_vector[index] * log(prediction) + (1.0 - target_vector[index]) * log(1.0 - prediction));
            }
            return loss / static_cast<double>(target_vector.size());
        }

        throw invalid_argument("Unsupported loss function: " + loss_function);
    }

    vector<double> loss_gradient(const vector<double>& predictions, const vector<double>& target_vector) const {
        if (predictions.size() != target_vector.size()) {
            throw invalid_argument("Prediction and target sizes must match.");
        }

        vector<double> gradient(predictions.size(), 0.0);
        const double epsilon = 1e-9;
        const double output_count = static_cast<double>(predictions.size());

        if (loss_function == "mean_squared_error") {
            for (size_t index = 0; index < predictions.size(); ++index) {
                gradient[index] = 2.0 * (predictions[index] - target_vector[index]) / output_count;
            }
            return gradient;
        }

        if (loss_function == "binary_cross_entropy") {
            for (size_t index = 0; index < predictions.size(); ++index) {
                double prediction = clamp(predictions[index], epsilon, 1.0 - epsilon);
                gradient[index] = (prediction - target_vector[index]) / (prediction * (1.0 - prediction) * output_count);
            }
            return gradient;
        }

        throw invalid_argument("Unsupported loss function: " + loss_function);
    }

    void backward_pass(const vector<double>& target_vector, double learning_rate) {
        vector<double> gradient = loss_gradient(output_layer.output_vector, target_vector);
        gradient = output_layer.backward(gradient, learning_rate);

        for (int layer_index = static_cast<int>(hidden_layers.size()) - 1; layer_index >= 0; --layer_index) {
            gradient = hidden_layers[layer_index].backward(gradient, learning_rate);
        }
    }

    void train(
        const vector<vector<double>>& input_data,
        const vector<vector<double>>& target_data,
        int epochs,
        double learning_rate,
        int log_interval = 200
    ) {
        if (input_data.empty() || input_data.size() != target_data.size()) {
            throw invalid_argument("Input and target datasets must be non-empty and same size.");
        }
        if (epochs <= 0 || learning_rate <= 0.0) {
            throw invalid_argument("Epochs and learning rate must be positive.");
        }

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double epoch_loss = 0.0;

            for (size_t sample_index = 0; sample_index < input_data.size(); ++sample_index) {
                const vector<double>& sample_input = input_data[sample_index];
                const vector<double>& sample_target = target_data[sample_index];

                vector<double> prediction = forward_pass(sample_input);
                epoch_loss += calculate_loss(prediction, sample_target);
                backward_pass(sample_target, learning_rate);
            }

            epoch_loss /= static_cast<double>(input_data.size());
            if (epoch == 1 || epoch % log_interval == 0 || epoch == epochs) {
                cout << "Epoch " << epoch << " - Loss: " << fixed << setprecision(6) << epoch_loss << endl;
            }
        }
    }

    vector<double> predict(const vector<double>& input) {
        return forward_pass(input);
    }
};

int main() {
    vector<vector<double>> xor_inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    vector<vector<double>> xor_targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    NeuralNetwork network(
        2,
        2,
        4,
        1,
        "tanh",
        "sigmoid",
        "binary_cross_entropy"
    );

    network.train(xor_inputs, xor_targets, 5000, 0.1, 500);

    cout << "\nPredictions after training:" << endl;
    for (const vector<double>& sample : xor_inputs) {
        vector<double> prediction = network.predict(sample);
        cout << sample[0] << " XOR " << sample[1] << " -> " << fixed << setprecision(6) << prediction[0] << endl;
    }

    return 0;
}
