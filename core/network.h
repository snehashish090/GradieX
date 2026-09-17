/*
    @file core/network.h
    @brief Defines the Network class for managing layers, forward and backward passes, and gradient updates.

    @author Snehashish Laskar

    The architecture:
        - The network consists of multiple hidden layers followed by an output layer.
        - Each layer is represented by either a HiddenLayer or OutputLayer object.
        - The network manages forward and backward passes, gradient accumulation, and weight updates.
        - Hidden layers and output layers are stored in separate vectors for efficient access and management.

    Memory Management:
        - The network uses vectors of pointers to manage hidden and output layers.
        - Each layer is dynamically allocated and deallocated in the Network destructor.
        - This ensures contiguous memory allocation for the vectors themselves while allowing flexible layer management.
    
    The math behind this architecture is included under math.tex and math.pdf
*/
#pragma once
#include "layer.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace core
{
    class Network
    {
        // Network configuration and layers management
        public:
        int numHiddenLayers;
        int numOutputLayers = 1; // Assumes 1 hidden layer minimum
        float learningRate = 0.01f;
        int batches = 1;

        std::vector<HiddenLayer*> hiddenLayers;
        std::vector<OutputLayer*> outputLayers;

        // A default-constructed network owns no layers, so both counts read 0. Claiming
        // one output layer here would leave numOutputLayers disagreeing with
        // outputLayers.size(), and anything trusting the count (accumulateGradients
        // indexes outputLayers[0]) would read past the end of an empty vector.
        Network() : numHiddenLayers(0), numOutputLayers(0), learningRate(0.01f), batches(1) {}
        // We use MIL here LOL. Guess I am not old fashioned >_< (read layer.h if you dont follow my joke)
        Network(int numHiddenLayers, int numOutputLayers, float learningRate, int batches = 1)
            : numHiddenLayers(numHiddenLayers), numOutputLayers(numOutputLayers),
              learningRate(learningRate), batches(batches)
        {
            // Resize the vectors to hold the specified number of hidden and output layers
            // We do this so there is no heap allocations
            hiddenLayers.resize(numHiddenLayers);
            outputLayers.resize(numOutputLayers);
        }

        void addHiddenLayer(
            int layerIndex,
            int width, int inputSize,
            core::functions::activations::ActivationFunction activation,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss
        )
        {
            /**
             * @brief Adds a hidden layer at the specified index with the given configuration.
             * @param layerIndex The index at which to add the hidden layer.
             * @param width The number of neurons in the hidden layer.
             * @param inputSize The size of the input vector to the hidden layer.
             * @param activation The activation function for the hidden layer.
             * @param init The weight initializer function.
             * @param loss The loss function for the hidden layer.
             */
            // operator[] here would write past the end of the vector and leak the layer
            // we just allocated, so the index is checked before anything is allocated.
            if (layerIndex < 0 || layerIndex >= static_cast<int>(hiddenLayers.size()))
                throw std::out_of_range(
                    "addHiddenLayer: layerIndex " + std::to_string(layerIndex) +
                    " is outside the " + std::to_string(hiddenLayers.size()) + " hidden slots");

            HiddenLayer* layer = new HiddenLayer(
                width, inputSize,
                activation,
                init, loss
            );
            delete hiddenLayers[layerIndex];   // replacing a filled slot must not leak
            hiddenLayers[layerIndex] = layer;
        }

        void addOutputLayer(
            int layerIndex,
            int width, int inputSize,
            core::functions::activations::ActivationFunction activation,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss
        )
        {
            /**
             * @brief Adds an output layer at the specified index with the given configuration.
             * @param layerIndex The index at which to add the output layer.
             * @param width The number of neurons in the output layer.
             * @param inputSize The size of the input vector to the output layer.
             * @param activation The activation function for the output layer.
             * @param init The weight initializer function.
             * @param loss The loss function for the output layer.
             */
            if (layerIndex < 0 || layerIndex >= static_cast<int>(outputLayers.size()))
                throw std::out_of_range(
                    "addOutputLayer: layerIndex " + std::to_string(layerIndex) +
                    " is outside the " + std::to_string(outputLayers.size()) + " output slots");

            OutputLayer* layer = new OutputLayer(
                width, inputSize,
                activation,
                init, loss
            );
            delete outputLayers[layerIndex];   // replacing a filled slot must not leak
            outputLayers[layerIndex] = layer;
        }

        void initializeAllWeightsAndBiases()
        {
            /**
             * @brief Initializes the weights and biases for all layers in the network.
             * 
             * Why do we not use SIMD instructions for initializing weights and biases?
             * Because the underlying functions rely on std::random and other non-SIMD-friendly operations.
             */
            // Slots left unfilled are null; the destructor already tolerates them, so
            // this loop does too rather than dereferencing a null layer.
            for (auto& layer : hiddenLayers)
                if (layer != nullptr) layer->initializeWeightsAndBiases();
            for (auto& layer : outputLayers)
                if (layer != nullptr) layer->initializeWeightsAndBiases();
        }

        ~Network()
        {
            /**
             * @brief Destructor for the Network class. Deletes all hidden and output layers to free memory.
             * @note This is important to prevent memory leaks, as the layers are dynamically allocated.
             */
            for (auto& layer : hiddenLayers)
                delete layer;
            for (auto& layer : outputLayers)
                delete layer;
        }

        void forwardPass(const float* inputs)
        {
            /**
             * @brief Performs a forward pass through the network, calculating the outputs for all layers.
             * @param inputs The input vector to the network. (So per epoch basis the inputs will be passed)
             *  The inputs are passed through pointers so we need the inputs to survive the stack during the whole process
             */
            const float*__restrict__ currentInput = inputs;
            // Just sequationally call the calculateOutputs function for each layer
            // We cannot parallelize this easily because each layer depends on the output of the previous layer.
            // Thus no SIMD either
            for (auto& layer : hiddenLayers)
            {
                layer->calculateOutputs(currentInput);
                currentInput = layer->neuronOutputsActivated.data();
            }
            for (auto& layer : outputLayers)
            {
                layer->calculateOutputs(currentInput);
                currentInput = layer->neuronOutputsActivated.data();
            }
        }

        void zeroAllGradients()
        {
            /**
             * @brief Zeros out all gradients for all layers in the network.
             * This is typically done at the beginning of a new training iteration to prevent 
             * accumulation of gradients from previous iterations.
             */
            for (auto& layer : hiddenLayers)
                layer->zeroGradients();
            for (auto& layer : outputLayers)
                layer->zeroGradients();
        }

        void scaleAllGradients(float scale)
        {
            /**
             * @brief Scales all gradients for all layers in the network by the given factor.
             * @param scale The factor by which to scale the gradients.
             */
            for (auto& layer : hiddenLayers)
                layer->scaleGradients(scale);
            for (auto& layer : outputLayers)
                layer->scaleGradients(scale);
        }

        void applyGradients()
        {
            /**
             * @brief Applies the accumulated gradients to update the weights and biases of all layers in the network.
             * This is typically done after computing and possibly scaling the gradients.
             */
            for (auto& layer : hiddenLayers)
                layer->updateWeightsAndBiases(this->learningRate);
            for (auto& layer : outputLayers)
                layer->updateWeightsAndBiases(this->learningRate);
        }

        void accumulateGradients(std::vector<float>& targetOutputs)
        {
            /**
             * @brief Accumulates gradients for all layers in the network based on the provided target outputs.
             * @param targetOutputs The target output vectors for the current batch.
             * 
             * We have moved from the older versions as we now batch and accumalate our gradients. 
             * Why? 
             * Accumulating gradients over a batch helps in stabilizing the training process and can lead
             * to better convergence compared to updating weights after every single sample. [form my research on this topic]
             */
            std::vector<float>& currentTarget = targetOutputs;

            for (int i = this->numOutputLayers - 1; i >= 0; i--)
            {
                // Go through the output layers, this will in 99% cases just be a single layer
                OutputLayer* currentLayer = this->outputLayers[i];
                currentLayer->computeErrorSignal(currentTarget.data());
                currentLayer->computeGradients();
            }

            // Transfer the results from the output layer into aliases 
            // These aliases will be used to propagate the error signals backward through the hidden layers.
            int nextLayerWidth = this->outputLayers[0]->layerWidth;
            const float* nextLayerDeltas = this->outputLayers[0]->neuronDeltas.data();
            const float* nextLayerWeights = this->outputLayers[0]->neuronWeights.data();

            // Go through the hidden layers in reverse order to propagate the error signals backward.
            for (int i = this->numHiddenLayers - 1; i >= 0; i--)
            {   
                // Current hidden layer to process
                HiddenLayer* currentLayer = this->hiddenLayers[i];
                currentLayer->computeErrorSignal(nextLayerWidth, nextLayerDeltas, nextLayerWeights);
                currentLayer->computeGradients();

                // Store for next iterations
                nextLayerWidth = currentLayer->layerWidth;
                nextLayerDeltas = currentLayer->neuronDeltas.data();
                nextLayerWeights = currentLayer->neuronWeights.data();
            }
        }

        float computeLoss(const std::vector<float>& targetOutputs) const
        {
            /**
             * @brief Computes the loss of the network for the given target outputs.
             * @param targetOutputs The expected output values for the current input.
             * @return The computed loss value.
             */
            float totalLoss = 0.0f;
            size_t targetOffset = 0;

            // Compute the total loss by summing the loss from each output layer. [Usually just one output layer]
            for (const auto& layer : outputLayers)
            {
                totalLoss += layer->lossFunction(
                    targetOutputs.data() + targetOffset,
                    layer->neuronOutputsActivated.data(),
                    static_cast<size_t>(layer->layerWidth)
                );
                targetOffset += layer->layerWidth;
            }
            // Return the average loss over all output layers. [In most cases, this is just the loss of the single output layer]
            return totalLoss;
        }

        float trainBatch(const size_t* __restrict__ sampleOrder, size_t startSample, int batchSize,
                         std::vector<std::vector<float>>& inputs,
                         std::vector<std::vector<float>>& targetOutputs)
        {
            /**
             * @brief Trains the network on a batch of samples.
             * @param sampleOrder The order samples are visited in; sampleOrder[k] is the index
             *                    into inputs/targetOutputs of the k-th sample of the epoch.
             *                    Pass nullptr to walk the samples in their natural order.
             * @param startSample The position in sampleOrder of the first sample in the batch.
             * @param batchSize The number of samples in the batch.
             * @param inputs The input data for all samples.
             * @param targetOutputs The expected output values for all samples.
             * @return The average loss over the batch.
             */

            // Zero out all gradients before processing the batch to ensure fresh accumulation.
            zeroAllGradients();
            
            float batchLoss = 0.0f;
            for (int b = 0; b < batchSize; ++b)
            {
                // For every batch: perform forward pass, accumulate gradients, and compute loss.
                const size_t position = startSample + static_cast<size_t>(b);
                const size_t idx = (sampleOrder != nullptr) ? sampleOrder[position] : position;
                this->forwardPass(inputs[idx].data());
                this->accumulateGradients(targetOutputs[idx]);
                batchLoss += this->computeLoss(targetOutputs[idx]);
            }
            // After processing all samples in the batch, scale the accumulated gradients and apply them to update the network's parameters.
            // Scaling is necessary to ensure that the gradient updates are averaged over the batch rather than summed,
            // which helps maintain stable learning rates.
            this->scaleAllGradients(1.0f / static_cast<float>(batchSize));
            this->applyGradients();

            // Return the average loss over the batch.
            return batchLoss / static_cast<float>(batchSize);
        }

        float trainBatch(size_t startSample, int batchSize,
                         std::vector<std::vector<float>>& inputs,
                         std::vector<std::vector<float>>& targetOutputs)
        {
            /**
             * @brief Trains on a batch of samples in their natural order.
             * Convenience overload for callers that do not shuffle.
             */
            return trainBatch(nullptr, startSample, batchSize, inputs, targetOutputs);
        }

        void train(std::vector<std::vector<float>>& inputs,
                   std::vector<std::vector<float>>& targetOutputs,
                   int numEpochs, float* __restrict__ lossBuffer, int bufferSize,
                   bool verbose = false, bool shuffle = false,
                   std::optional<unsigned> seed = std::nullopt, int logEvery = 1)
        {
            /**
             * @brief Trains the network for a number of epochs over the whole dataset.
             * @param inputs One row per sample.
             * @param targetOutputs One row per sample, aligned with inputs.
             * @param numEpochs How many passes over the dataset to make.
             * @param lossBuffer Receives the mean loss of each epoch; must hold numEpochs floats.
             * @param bufferSize The capacity of lossBuffer, checked against numEpochs.
             * @param verbose Whether to print progress to stdout.
             * @param shuffle Whether to reshuffle the sample order at the start of every epoch.
             * @param seed Seed for the shuffle RNG. Omitted means non-deterministic.
             * @param logEvery Print every logEvery-th epoch when verbose (the last is always printed).
             *
             * Shuffling lives here rather than in the caller so that every training loop --
             * C++ or Python -- gets the same batching, the same gradient accumulation and the
             * same update rule. The only allocation is the sample-order vector, made once
             * before the first epoch; the per-epoch hot path stays allocation-free.
             */

            // Validate the input arguments to ensure consistency and prevent runtime errors.
            if (inputs.size() != targetOutputs.size())
                throw std::invalid_argument("train: inputs and targetOutputs must hold the same number of samples");
            if (inputs.empty())
                return;
            if (numEpochs <= 0)
                return;
            if (lossBuffer == nullptr || bufferSize < numEpochs)
                throw std::invalid_argument("train: lossBuffer must have at least 'numEpochs' elements and not be null");

            // Determine the number of samples and the batch size for training.
            const size_t numSamples = inputs.size();
            const int batchSize = (this->batches > 0) ? this->batches : static_cast<int>(numSamples);
            if (logEvery <= 0)
                logEvery = 1;

            // The visiting order for one epoch. Identity unless we shuffle.
            std::vector<size_t> sampleOrder(numSamples);
            std::iota(sampleOrder.begin(), sampleOrder.end(), size_t{0});
            std::mt19937 rng(seed.has_value() ? *seed : std::random_device{}());
            const size_t* __restrict__ order = sampleOrder.data();

            // Loop over each epoch and train the network on all batches of samples.
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                // A fresh order each epoch decorrelates the batches from the dataset's own ordering.
                if (shuffle)
                    std::shuffle(sampleOrder.begin(), sampleOrder.end(), rng);

                // Initialize the loss for the current epoch.
                float epochLoss = 0.0f;

                // Process all samples in batches.
                size_t sample = 0;
                while (sample < numSamples)
                {
                    const int thisBatch = static_cast<int>(
                        std::min<size_t>(static_cast<size_t>(batchSize), numSamples - sample)
                    );
                    const float batchLoss = trainBatch(order, sample, thisBatch, inputs, targetOutputs);

                    epochLoss += batchLoss * static_cast<float>(thisBatch);
                    sample += static_cast<size_t>(thisBatch);
                }

                // Store and display the average loss for the current epoch.
                const float meanLoss = epochLoss / static_cast<float>(numSamples);
                lossBuffer[epoch] = meanLoss;
                if (verbose && (epoch % logEvery == 0 || epoch == numEpochs - 1))
                    std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs
                              << ", Loss: " << meanLoss << std::endl;
            }
        }
    };
}