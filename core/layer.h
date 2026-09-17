/*

    @file core/layer.h
    @brief Defines the Layer class as a base structure for all neural network layers.

    @author Snehashish Laskar

    The architecture:
        - Earlier versions (v0.0.1) used an OOP style Neuron->Layer->Network hierarchy.
        - The current version (v0.0.2) uses a more efficient flat structure for layers. 
        - Weights and biases of an entire layer are stored in a flat 1D vector. Why? :
            A 2D vector like vector<vector<float>> is non-contiguous in memory and thus less cache-friendly. 
            Using a flat 1D vector ensures contiguous memory allocation, improving performance.
        - All neurons consistently recieve the same input vector, thus the input vector can be shared across all neurons,
            reducing memory overhead and improving cache efficiency.
        - Assuming n neurons or a layer "width" of n, the weight matrix will have dimensions n x inputSize, stored as a flat 1D vector.
        Weights Memory Layout : [w11, w12, w13, ..., w1inputSize, w21, w22, w23, ..., w2inputSize, ..., wn1, wn2, wn3, ..., wninputSize]
        - To access the weight of the i-th neuron for the j-th input, use the formula: weightIndex = i * inputSize + j.
        - The biases for each neuron are stored in a separate 1D vector, with the i-th neuron's bias at index i.    
        Biases Memory Layout : [b1, b2, b3, ..., bn]
    
    Memory Management:
        - Using flat 1D vectors for weights and biases ensures contiguous memory allocation.
        - The use of function pointers, pointers to vector data allows us to perform operations directly on the underlying memory
            through the "vectorScratchPad" you notice in the classes, reducing overhead and improving performance.
        - We aim for 0 heap allocations during the forward and backward passes by using pre-allocated scratch pads and flat vectors.
        - constexpr for constant values to enable compile-time evaluation and further reduce runtime overhead.
*/

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
        /**
            Layer class represents a single layer in a neural network.

            Code Design : 
                - pointers to vector data come in <type>* __restrict__ for better performance and to enable compiler optimizations.
                - When writing to vector data in for loops, we establish __restrict__ pointer first 
                    eg : vector[i] = data WRONG DO NOT DO THIS 
                    
                    Correct way:
                    float* __restrict__ vec = vector.data();
                    vec[i] = data;

                    Why?  Using __restrict__ informs the compiler that the pointer is the only reference to that memory. 
        */
        // Class Members
        public:

        int layerWidth = 0; // Number of neurons in the layer
        int inputSize=0;
        std::vector<float> layerInputs;
        core::functions::activations::ActivationFunction activationFunction = nullptr; // Activation function for the layer
        core::functions::activations::ActivationFunctionDerivative activationFunctionDerivative = nullptr; // Activation function derivative for the layer
        core::functions::initializers::initFunction initialiser = nullptr; // Initialiser for the layer weights
        core::functions::loss::LossFunction lossFunction = nullptr; // Loss function for the layer
        core::functions::loss::LossFunctionDerivative lossFunctionDerivative = nullptr; // Loss function derivative for the layer
        std::vector<float> neuronOutputsUnactivated; // Stores the z values
        std::vector<float> neuronOutputsActivated; // Stores the activated outputs (a values)
        std::vector<float> neuronDeltas; // Stores the error signals (delta values)
        std::vector<float> neuronWeights; // Stores the weights of the layer as a 1D Matrix
        // Why? a vector of vector causes heap allocations.
        std::vector<float> neuronBiases; // Stores the biases of the layer
        std::vector<float> scratchPad1; // Temporary storage for intermediate calculations
        std::vector<float> scratchPad2;
        std::vector<float> neuronWeightGradients; // Stores the gradients of the neurons
        std::vector<float> neuronBiasGradients; // Stores the gradients of the neuron biases

        Layer() noexcept = default;
        Layer(int width, 
            int inputSize,
            core::functions::activations::ActivationFunction activation,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss)
        {
            /**
             * Constructs a layer with the specified width, input size, activation function, initializer, and loss function. 
             * @param width The width of the layer
             * @param inputSize The size of the input vector
             * @param activation The activation function for the layer
             * @param init The initializer function for the layer's weights
             * @param loss The loss function associated with the layer
             */

            // Sorry that I am not using MIL :( Maybe I am old fashioned that way?
            this->layerWidth = width;
            this->inputSize = inputSize;
            this->activationFunction = activation;
            // The derivatved is DERIVED from the activation function (lmao :D)
            this->activationFunctionDerivative = core::functions::activations::fetchAssociatedDerivative(activation);
            this->lossFunctionDerivative = core::functions::loss::fetchAssociatedDerivative(loss);
            this->initialiser = init;
            this->lossFunction = loss;
            
            // Reserve memory for all the vectors 
            this->layerInputs.resize(inputSize);
            this->neuronOutputsUnactivated.resize(width);
            this->neuronOutputsActivated.resize(width);
            this->neuronDeltas.resize(width);
            this->neuronWeights.resize(width * inputSize);
            this->neuronBiases.resize(width);
            this->neuronWeightGradients.resize(width * inputSize);
            this->neuronBiasGradients.resize(width);

            this->scratchPad1.resize(width); // Allocate enough buffer
            this->scratchPad2.resize(width); // Allocate enough buffer
        }

        void initializeWeightsAndBiases()
        {
            /**
             * Initializes the weights and biases of the layer.
             *
             * @param neuronWeights Pointer to the neuron's weights
             * @param inputSize The size of the input vector
             * @param layerWidth The width of the layer
             */
            
            float* __restrict__ neuronWeights = this->neuronWeights.data();
            float* __restrict__ neuronBiases = this->neuronBiases.data();

            // Initialize the weights and biases using the provided initialiser
            this->initialiser(
                neuronWeights, // The neuron's weights
                this->inputSize,
                this->layerWidth
            );
            // Set the biases to zero initially {that is the general convention}
            for (int i = 0; i < layerWidth; ++i)
                neuronBiases[i] = 0.0f;
        }

        void calculateOutputs(const float* __restrict__ input)
        {
            /**
             * Calculates the outputs of the layer given the input vector.
             * Vector Equation :  z^{[l]} = W^{[l]} * a^{[l-1]} + b^{[l]}
             * where 
             * z^{[l]} represents the unactivated outputs of the current layer
             * W^{[l]} represents the weight matrix, a^{[l-1]} represents the activated outputs of the previous layer
             * b^{[l]} represents the biases of the current layer.
             * @param input Pointer to the input vector
             * @return void
             */

            float* __restrict__ layerInputs = this->layerInputs.data();
            float* __restrict__ neuronOutputsUnactivated = this->neuronOutputsUnactivated.data();
            float* __restrict__ neuronOutputsActivated = this->neuronOutputsActivated.data();
            const float* __restrict__ neuronWeights = this->neuronWeights.data();
            const float* __restrict__ neuronBiases = this->neuronBiases.data();

            const int width     = this->layerWidth; // Load ahead of time
            const int inputSize = this->inputSize;

            for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                // Cache the inputs passed into the layer's member variable
                layerInputs[inputIdx] = input[inputIdx];
            
            for (int neuron = 0; neuron < width; ++neuron)
            {
                const float* __restrict__ w = neuronWeights + neuron * inputSize;
                float sum = 0.0f;
                // Vectorise the dot product computation using OpenMP SIMD directive
                #pragma omp simd reduction(+:sum)
                for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                    sum += w[inputIdx] * layerInputs[inputIdx];
                neuronOutputsUnactivated[neuron] = sum + neuronBiases[neuron];
            }
            // Activate the output. Is this a trivial step? yes. Can it be parameterized? Yes. 
            // Why is not optional? Because I wanna build in baby steps. Let's not get ahead of ourselves. 
            // And focus on the major features I need to build this project :D
            this->activationFunction(neuronOutputsUnactivated,
                                    neuronOutputsActivated,
                                    width);
        }

        void zeroGradients()
        {
            /**
             * Zeros out the gradients of the layer's weights and biases.
             * @return void
             * This is typically done at the beginning of a training step to ensure that gradients from previous steps do not accumulate.
             */
            // Set our local pointers to the vectors for easier access and potential SIMD optimization.
            float* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            float* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            const int totalWeights = this->layerWidth * this->inputSize;
            const int width        = this->layerWidth;
            // I think its pretty simple, just set all gradients & biases to zero.
            #pragma omp simd
            for (int i = 0; i < totalWeights; ++i)
                neuronWeightGradients[i] = 0.0f;

            #pragma omp simd
            for (int i = 0; i < width; ++i)
                neuronBiasGradients[i] = 0.0f;
        }

        void scaleGradients(float scale)
        {
            /**
             * Scales the gradients of the layer's weights and biases by the given factor.
             * @param scale The factor by which to scale the gradients.
             * @return void
             * 
             * Why would we want to scale gradients? 
             * This is typically done to adjust the learning rate dynamically or to implement gradient clipping.
             */
            float* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            float* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            const int totalWeights = this->layerWidth * this->inputSize;
            const int width        = this->layerWidth;

            // Again, we use SIMD to efficiently scale all gradients in parallel.
            // And we simply multiply vector elements by the scale factor.
            #pragma omp simd
            for (int i = 0; i < totalWeights; ++i)
                neuronWeightGradients[i] *= scale;

            #pragma omp simd
            for (int i = 0; i < width; ++i)
                neuronBiasGradients[i] *= scale;
        }

        void computeGradients()
        {
            /**
             * @brief Computes the gradients of the layer's weights and biases based on the current neuron deltas and inputs.
             * We will typically call this during the backpropagation step of training.
             * The computed gradients are stored internally and later used to update the weights and biases.
             * @return void
             * This function does not take any parameters as it operates on the internal state of the layer.
             */
            // Set vector pointers 
            const float* __restrict__ savedInputs = this->layerInputs.data();
            const float* __restrict__ neuronDeltas = this->neuronDeltas.data();
            float* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            float* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            const int width = this->layerWidth;
            const int inputSize = this->inputSize;

            // Compute the gradients for each neuron in the layer.
            // Why do we not optimise for SIMD? 
            // The outer loop iterates over neurons, and each neuron's gradient computation depends on the corresponding delta. 
            // Causing dependencies between iterations of the outer loop, which prevents effective SIMD parallelization.
            // SIMD is applied to the inner loop over inputs, 
            // which is where the bulk of the computation occurs and can benefit from vectorization.
            for (int neuron = 0; neuron < width; ++neuron)
            {
                const float delta = neuronDeltas[neuron];
                float* __restrict__ dW = neuronWeightGradients + neuron * inputSize;

                #pragma omp simd
                for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                    dW[inputIdx] += delta * savedInputs[inputIdx];

                neuronBiasGradients[neuron] += delta;
            }
        }

        void updateWeightsAndBiases(float learningRate)
        {
            /**
             * @brief Updates the weights and biases of the layer using the computed gradients.
             * @param learningRate The learning rate to scale the gradient updates.
             * @return void
             * This function updates the internal weights and biases of the layer based on the previously computed gradients.
             */
            const float* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            const float* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            float* __restrict__ neuronWeights = this->neuronWeights.data();
            float* __restrict__ neuronBiases  = this->neuronBiases.data();
            const int width = this->layerWidth;
            const int inputSize = this->inputSize;

            // Again, we apply SIMD only to the inner loop over inputs for efficient vectorized updates.
            for (int neuron = 0; neuron < width; ++neuron)
            {
                #pragma omp simd
                for (int input = 0; input < inputSize; ++input)
                    neuronWeights[neuron * inputSize + input] -= 
                    learningRate * neuronWeightGradients[neuron * inputSize + input];
                neuronBiases[neuron] -= learningRate * neuronBiasGradients[neuron];
            }
        }
    };

    class OutputLayer : public Layer
    {
        public:
            using Layer::Layer;
            void computeErrorSignal(const float*__restrict__ target)
            {
                /**
                 * @brief Computes the error signal (deltas) for the output layer based on the target output.
                 * @param target Pointer to the array of target output values.
                 * @return void
                 * This function calculates the error signal for each neuron in the output layer by computing the derivative of 
                 * the loss function with respect to the neuron's activated output and multiplying it by the derivative of the 
                 * activation function.
                 */
                float* neuronDeltas = this->neuronDeltas.data();
                // Compute the derivative of the loss function with respect to the activated outputs.
                float* lossDerivative = this->lossFunctionDerivative(
                    target,
                    this->neuronOutputsActivated.data(),
                    scratchPad1.data(),
                    this->layerWidth
                );
                // Compute the derivative of the activation function with respect to the unactivated outputs.
                float* activationDerivedInputs = this->activationFunctionDerivative(
                    this->neuronOutputsUnactivated.data(),
                    scratchPad2.data(),
                    this->layerWidth
                );
                #pragma omp simd 
                // Compute the error signal for each neuron in the output layer.
                // Which is the element-wise product of the loss derivative and the activation function derivative.
                for (int neuron = 0; neuron < this->layerWidth; ++neuron)
                    neuronDeltas[neuron] = lossDerivative[neuron] * activationDerivedInputs[neuron];
            }
    };
    
    // Has a different way of calculating error signal
    class HiddenLayer : public Layer
    {
        /**
         * What is different? 
         * The HiddenLayer computes its error signal based on the deltas and weights of the next layer, 
         * rather than directly from the target output as in the OutputLayer.
         */
    public:
        using Layer::Layer;
        void computeErrorSignal(int nextLayerWidth, 
            const float* nextLayerDeltas, 
            const float* nextLayerWeights)
        {
            /**
             * @brief Computes the error signal (deltas) for the hidden layer based on the next layer's deltas and weights.
             * @param nextLayerWidth The number of neurons in the next layer.
             * @param nextLayerDeltas Pointer to the array of deltas from the next layer.
             * @param nextLayerWeights Pointer to the array of weights connecting this layer to the next layer.
             * @return void
             * This function calculates the error signal for each neuron in the hidden layer by propagating the errors from 
             * the next layer backward through the weights and applying the derivative of the activation function.
             */
            float* neuronDeltas = this->neuronDeltas.data();

            // 1. Compute derivatives for the whole layer in one go.
            float* activationDerivedInputs = this->activationFunctionDerivative(
                this->neuronOutputsUnactivated.data(),
                scratchPad2.data(),
                this->layerWidth
            );

            const int width = this->layerWidth;

            #pragma omp simd
            // Empty before insertion of propagated errors from the next layer.
            for (int neuron = 0; neuron < width; ++neuron)
                neuronDeltas[neuron] = 0.0f;

            for (int nextNeuron = 0; nextNeuron < nextLayerWidth; ++nextNeuron)
            {
                // Get the delta from the l+1 (next) layer
                const float nextDelta = nextLayerDeltas[nextNeuron];
                const float* __restrict__ nextWeightRow = nextLayerWeights + static_cast<size_t>(nextNeuron) * width;
                #pragma omp simd
                // Propagate the error from the next layer to the current layer.
                for (int neuron = 0; neuron < width; ++neuron)
                    neuronDeltas[neuron] += nextDelta * nextWeightRow[neuron]; 
            }

            #pragma omp simd
            for (int neuron = 0; neuron < width; ++neuron)
                // Multiply by the derivative of the activation function
                neuronDeltas[neuron] *= activationDerivedInputs[neuron];
        }
    };

    
}