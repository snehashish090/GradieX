#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  
#include <ctime>
#include <string>
using namespace std;

class Neuron
{
    public:
        vector<double> weights; // weight vectors 
        vector<double> x; // input vectors
        double bias;
        double output; // output of the neuron
        int feature_count;

        Neuron(int sample_size){
            // Initialization function
            feature_count = sample_size;
            for (int i=0; i<sample_size; i++){
                weights.push_back(0.0); 
                x.push_back(0.0); 
            }
            bias = 0.0; 
            output = 0.0;
        }

        Neuron(double bias, vector<double> weights, vector<double> input_vector){
            this->bias = bias;
            this->weights = weights;
            this->x = input_vector;
            this->output = 0.0;
            this->feature_count = input_vector.size();
        }

        double sigmoid_activation(double sum) {
            if (sum >= 0){return 1.0 / (1.0 + exp(-sum));} 
            else {return  1.0 / (1.0 + exp(sum));}
        } 
        double hyperbolic_tangent_activation(double sum){
           return (exp(sum) - exp(-sum)) / (exp(sum) + exp(-sum));
        }
        double ReLU_activation(double sum){return max(0.0, sum);}
        double linear_activation(double sum){return sum;}
        double leaky_ReLU_activation(double sum){return max(0.01 * sum, sum);}
        double softmax_activation(double sum, vector<double> output_vector){
            double exp_sum = exp(sum);
            double exp_total = 0.0;

            for (double output : output_vector){
                exp_total += exp(output);
            }
            return exp_sum / exp_total;
        }

        double calculate_output(string activation_function){
            double sum = bias;
            for (int i=0; i<feature_count; i++){
                sum += weights[i] * x[i]; 
            }
            if (activation_function == "sigmoid"){
                output = sigmoid_activation(sum);
            } else if (activation_function == "tanh"){
                output = hyperbolic_tangent_activation(sum);
            } else if (activation_function == "ReLU"){
                output = ReLU_activation(sum);
            } else if (activation_function == "linear"){
                output = linear_activation(sum);
            } else if (activation_function == "leaky_ReLU"){
                output = leaky_ReLU_activation(sum);
            } else {
                cout << "Invalid activation function specified." << endl;
            }
            return output;
        }
};

class Layer
{
    public:
        vector<Neuron> neurons; // vector of neurons in the layer
        int neuron_count;
        string activation_function;
        vector<double> output_vector; // input vector for the layer

        Layer(int neuron_count, int sample_size, string activation_function){
            this->neuron_count = neuron_count;
            this->activation_function = activation_function;
            for (int i=0; i<neuron_count; i++){
                neurons.push_back(Neuron(sample_size)); 
            }
        }

        Layer(int neuron_count, vector<double> input_vector, vector<double> bias, 
            vector<vector<double>> weights, string activation_function){
            this->neuron_count = neuron_count;
            this->activation_function = activation_function;
            
            for (int k=0; k<neuron_count; k++){
                neurons.push_back(Neuron(bias[k], weights[k], input_vector)); 
            }
        }

        vector<double> calculate_output_vector(){
            output_vector.clear();
            for (int i=0; i<neuron_count; i++){
                output_vector.push_back(neurons[i].calculate_output(activation_function));
            }
            return output_vector;
        }
};

vector<vector<double>> initialize_weights(int layer_width, int input_size){
    vector<vector<double>> weights;
    for (int k=0; k<layer_width; k++){
        vector<double> weight_vector;
        for (int j=0; j<input_size; j++){
            weight_vector.push_back( (rand() / (double)RAND_MAX - 0.5) * 0.2);
        }
        weights.push_back(weight_vector);
    }
    return weights;
}

class NeuralNetwork
{
    public:
        vector<Layer> hidden_layers; // vector of layers in the neural network
        int hidden_layer_count;
        string hidden_layer_activation_function;
        string loss_function;
        vector<double> input_vector; 
        int layer_width;
        Layer output_layer;
        int output_layer_width;
        string output_activation_function;
        vector<double> desired_target_vector;
        vector<vector<double>> weights;

        NeuralNetwork(
            int hidden_layers, 
            int layer_width, 
            vector<double> input_vector, 
            string hidden_layer_activation_function,
            string output_activation_function,
            string loss_function, 
            int output_layer_width, 
            bool weight_initialization_required
        )
        {
            this->hidden_layer_count = hidden_layers;
            this->input_vector = input_vector;
            this->hidden_layer_activation_function = hidden_layer_activation_function;
            this->loss_function = loss_function;
            this->output_activation_function = output_activation_function;
            this->output_layer_width = output_layer_width;
            this->layer_width = layer_width;
        }

        vector<double> forward_pass(){
            vector<double> previous_input_vector = input_vector;
            for (int i = 0; i < hidden_layer_count; i++) {
                vector<double> bias_vector(layer_width, 0.0);
                vector<vector<double>> weights = initialize_weights(layer_width, previous_input_vector.size());
                hidden_layers.push_back(Layer(layer_width, previous_input_vector, bias_vector, 
                    weights, hidden_layer_activation_function));
                previous_input_vector = hidden_layers.back().calculate_output_vector();
            }

            output_layer = Layer(output_layer_width, previous_input_vector, 
                vector<double>(output_layer_width, 0.0), 
                initialize_weights(output_layer_width, previous_input_vector.size()), 
                output_activation_function
            );
            return output_layer.calculate_output_vector();
        }

        void calculate_loss(vector<double> target_vector, string loss_function)
        {
            double loss = 0.0;
            if (loss_function == "mean_squared_error"){
                for (int i=0; i<target_vector.size(); i++){
                    loss += pow((target_vector[i] - output_layer.output_vector[i]), 2);
                }
                loss /= target_vector.size();
            } else if (loss_function == "cross_entropy"){
                for (int i=0; i<target_vector.size(); i++){
                    loss -= target_vector[i] * log(output_layer.output_vector[i]);
                }
                loss /= target_vector.size();
            } else {
                cout << "Invalid loss function specified." << endl;
            }
            cout << "Loss: " << loss << endl;
        }
};


int main() {
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator  
    cout << "Hello, World!" << endl;
    return 0;
}
