#include <iostream>
#include <vector>
#include <cmath>
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

class Layer:
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
        Layer(int neuron_count, vector<vector<double>> input_vector, vector<double> bias, 
            vector<vector<double>> weights, string activation_function){
            this->neuron_count = neuron_count;
            this->activation_function = activation_function;
            
            for (int k=0; k<neuron_count; k++){
                neurons.push_back(Neuron(bias[k], weights[k], input_vector[k])); 
            }
        }
};

class NeuralNetwork
{
    public:
        vector<Layer> layers; // vector of layers in the neural network
        int layer_count;
        string activation_function;
        string loss_function;
        vector<double> input_vector; 

        NeuralNetwork(int layers, int layer_width, vector<double> input_vector, 
            string activation_function, string loss_function)
        {
            this->layer_count = layers;
            this->input_vector = input_vector;
            this->activation_function = activation_function;
            this->loss_function = loss_function;
        }

        void forward_pass(){
            for (int i = 0; i < layer_count; i++) {
                layers.push_back(Layer(layer_width, input_vector.size(), activation_function));
            }
        }      
};


int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
