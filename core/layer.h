#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include "activations.h"
#include "initializer.h"
#include "loss.h"
namespace core
{
    // Base structure for all layers
    class Layer
    {
        // Class Members
        public:
        int layerWidth = 0; // Number of neurons in the layer
        int inputSize=0;
        std::vector<double> layerInputs;
        core::functions::activations::ActivationFunction activationFunction; // Activation function for the layer
        core::functions::activations::ActivationFunctionDerivative activationFunctionDerivative; // Activation function derivative for the layer
        core::functions::initializers::initFunction initialiser; // Initialiser for the layer weights
        core::functions::loss::LossFunction lossFunction; // Loss function for the layer
        core::functions::loss::LossFunctionDerivative lossFunctionDerivative; // Loss function derivative for the layer
        std::vector<double> neuronOutputsUnactivated; // Stores the z values
        std::vector<double> neuronOutputsActivated; // Stores the activated outputs (a values)

        std::vector<double> neuronDeltas; // Stores the error signals (delta values)
        std::vector<double> neuronWeights; // Stores the weights of the layer as a 1D Matrix
        // Why? a vector of vector causes heap allocations.
        std::vector<double> neuronBiases; // Stores the biases of the layer
        std::vector<double> scratchPad1; // Temporary storage for intermediate calculations
        std::vector<double> scratchPad2;

        std::vector<double> neuronWeightGradients; // Stores the gradients of the neurons
        std::vector<double> neuronBiasGradients; // Stores the gradients of the neuron biases

        Layer() noexcept = default;
        Layer(int width, 
            int inputSize,
            core::functions::activations::ActivationFunction activation,
            core::functions::activations::ActivationFunctionDerivative activationDerivative,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss,
            core::functions::loss::LossFunctionDerivative lossDerivative)
        {
            // Rule : Be explicit and clear when assigning member variables
            this->layerWidth = width;
            this->inputSize = inputSize;
            this->activationFunction = activation;
            this->activationFunctionDerivative = activationDerivative;
            this->initialiser = init;
            this->lossFunction = loss;
            this->lossFunctionDerivative = lossDerivative;

            // Reserve memory for all the vectors 
            this->layerInputs.resize(inputSize);
            this->neuronOutputsUnactivated.resize(width);
            this->neuronOutputsActivated.resize(width);
            this->neuronDeltas.resize(width);
            this->neuronWeights.resize(width * inputSize);
            this->neuronBiases.resize(width);
            this->neuronWeightGradients.resize(width * inputSize);
            this->neuronBiasGradients.resize(width);

            this->scratchPad1.resize(inputSize*width); // Allocate enough buffer
            this->scratchPad2.resize(inputSize*width); // Allocate enough buffer
        }

        void initializeWeightsAndBiases()
        {
            // Initialize the weights and biases using the provided initialiser
            this->initialiser(
                this->neuronWeights.data(), // The neuron's weights
                this->inputSize,
                this->layerWidth
            );
            #pragma GCC unroll 4
            for (int i = 0; i < layerWidth; ++i)
                this->neuronBiases[i] = 0.0;
        }

        void calculateOutputs(const double*__restrict__ input)
        {
            // Keep the input around: computeGradients() needs it, since
            // dL/dW[neuron][i] = delta[neuron] * input[i]. Copied once here
            // rather than inside the neuron loop, which would rewrite the same
            // values layerWidth times.
            #pragma GCC unroll 4
            for (int inputIdx = 0; inputIdx < this->inputSize; ++inputIdx)
                this->layerInputs[inputIdx] = input[inputIdx];

            // Calculate the outputs based on the current inputs
            #pragma GCC unroll 4
            for (int neuron=0; neuron < this->layerWidth; neuron++)
            {
                double sum = 0.0;
                #pragma GCC unroll 4
                for (int inputIdx = 0; inputIdx < this->inputSize; ++inputIdx)
                    sum += this->neuronWeights[neuron * this->inputSize + inputIdx] * input[inputIdx];
                sum += this->neuronBiases[neuron];
                this->neuronOutputsUnactivated[neuron] = sum;
            }
            this->activationFunction(
                this->neuronOutputsUnactivated.data(),
                this->neuronOutputsActivated.data(),
                this->layerWidth
            );
        }

        void computeGradients()
        {
            std::vector<double>& savedInputs = this->layerInputs;
            for (int neuron = 0; neuron < this->layerWidth; ++neuron)
            {
                for (int inputIdx = 0; inputIdx < this->inputSize; ++inputIdx)
                    this->neuronWeightGradients[neuron * this->inputSize + inputIdx] = 
                    this->neuronDeltas[neuron] * savedInputs[inputIdx];
                this->neuronBiasGradients[neuron] = this->neuronDeltas[neuron];
            }
        }

        void updateWeightsAndBiases(double learningRate)
        {
            for (int neuron = 0; neuron < this->layerWidth; ++neuron)
            {
                for (int input = 0; input < this->inputSize; ++input)
                    this->neuronWeights[neuron * this->inputSize + input] -= 
                    learningRate * this->neuronWeightGradients[neuron * this->inputSize + input];
                this->neuronBiases[neuron] -= learningRate * this->neuronBiasGradients[neuron];
            }
        }

        void print() const 
        {
            std::cout << "Layer Width: " << this->layerWidth << "\n";
            std::cout << "Input Size: " << this->inputSize << "\n";
            std::cout << "Neuron Outputs (Unactivated): ";
            for (const auto& val : this->neuronOutputsUnactivated)
                std::cout << val << " ";
            std::cout << "\n";
            std::cout << "Neuron Outputs (Activated): ";
            for (const auto& val : this->neuronOutputsActivated)
                std::cout << val << " ";
            std::cout << "\n";
            std::cout << "Neuron Deltas: ";
            for (const auto& val : this->neuronDeltas)
                std::cout << val << " ";
            std::cout << "\n";
            std::cout << "Neuron Weights: \n";
            for (int i = 0; i < this->layerWidth; ++i) {
                for (int j = 0; j < this->inputSize; ++j)
                    std::cout << this->neuronWeights[i * this->inputSize + j] << " ";
                std::cout << "\n";
            }
            std::cout << "Neuron Biases: ";
            for (const auto& val : this->neuronBiases)
                std::cout << val << " ";
            std::cout << "\n";
        }
    };

    // Has a different way of calculating error signal
    class HiddenLayer : public Layer
    {
    public:
        using Layer::Layer;
        void computeErrorSignal(int nextLayerWidth, 
            const double* nextLayerDeltas, 
            const double* nextLayerWeights)
        {
            for (int neuron = 0; neuron < this->layerWidth; ++neuron)
            {
                double errorSum = 0.0;
                for (int nextNeuron = 0; nextNeuron < nextLayerWidth; ++nextNeuron)
                {
                    // Use this->layerWidth (the next layer's input size), NOT this->inputSize
                    errorSum += nextLayerDeltas[nextNeuron] * 
                        nextLayerWeights[nextNeuron * this->layerWidth + neuron];
                }

                this->neuronDeltas[neuron] = errorSum * this->activationFunctionDerivative(
                    &this->neuronOutputsUnactivated[neuron],
                    &scratchPad2[neuron],
                    1
                )[0];
            }
        }
    };

    class OutputLayer : public Layer
    {
        public:
            using Layer::Layer;
            void computeErrorSignal(const double*__restrict__ target)
            {
                double* lossDerivative = this->lossFunctionDerivative(
                    target,
                    this->neuronOutputsActivated.data(),
                    scratchPad1.data(),
                    this->layerWidth
                );
                double* activationDerivedInputs = this->activationFunctionDerivative(
                    this->neuronOutputsUnactivated.data(),
                    scratchPad2.data(),
                    this->layerWidth
                );
                for (int neuron = 0; neuron < this->layerWidth; ++neuron)
                    this->neuronDeltas[neuron] = lossDerivative[neuron] * activationDerivedInputs[neuron];
            }
    };
}