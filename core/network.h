/*
    @file core/network.h
    @brief Defines the Network class for managing layers, forward and backward passes, and gradient updates.

    @author Snehashish Laskar

    The architecture:
        - The network consists of multiple hidden layers followed by an output layer.
        - Each layer is represented by either a HiddenLayer or OutputLayer object.
        - The network manages forward and backward passes, gradient accumulation, and weight updates.
        - Hidden layers and output layers are stored in separate vectors for efficient access and management.

    v0.0.4 data types:
        - Network is a template on the IO scalar type (default float): the type of the
          vectors exchanged with the outside world -- forwardPass() inputs, train()/
          trainBatch() datasets, targets and the loss buffer.
        - Each *layer* picks its own dtype independently. addHiddenLayer()/addOutputLayer()
          take a DType argument alongside the activation/initializer/loss function pointers,
          and the network validates that the DType matches the scalar type of those pointers.
          Adjacent layers may differ (e.g. a float64 hidden layer feeding a float32 output
          layer); each layer converts incoming buffers into its own type at the boundary.
        - `Network` written without a template argument is `Network<float>` -- the float32
          network the engine has always been, and what the Python bindings use.

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
    template <typename IO = float>
    class Network
    {
        // Network configuration and layers management
        public:
        using Scalar = IO; // the type crossing the network boundary

        int numHiddenLayers;
        int numOutputLayers = 1; // Assumes 1 output layer minimum
        float learningRate = 0.01f;
        int batches = 1;

        // The dtype of every vector crossing the network boundary (inputs to
        // forwardPass, targets, datasets, loss buffer).
        static constexpr DType ioDType = dtype_of<IO>::value;

        std::vector<LayerBase*> hiddenLayers;
        std::vector<LayerBase*> outputLayers;

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
            hiddenLayers.resize(static_cast<size_t>(numHiddenLayers));
            outputLayers.resize(static_cast<size_t>(numOutputLayers));
        }

        // A network owns its layers; a memberwise copy would let two networks
        // delete the same LayerBase pointers.
        Network(const Network&) = delete;
        Network& operator=(const Network&) = delete;

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

        template <typename T>
        void addHiddenLayer(
            int layerIndex,
            int width, int inputSize,
            DType dtype,
            core::functions::activations::ActivationFunction<T> activation,
            core::functions::initializers::initFunction<T> init,
            core::functions::loss::LossFunction<T> loss
        )
        {
            /**
             * @brief Adds a hidden layer at the specified index with the given configuration.
             * @param layerIndex The index at which to add the hidden layer.
             * @param width The number of neurons in the hidden layer.
             * @param inputSize The size of the input vector to the hidden layer.
             * @param dtype The scalar type the layer computes in; must match the type of the
             *              activation/initializer/loss function pointers.
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

            // The runtime dtype argument must agree with the compile-time scalar type
            // of the functions being installed, otherwise the layer would cast between
            // them and compute garbage.
            if (dtype != dtype_of<T>::value)
                throw std::invalid_argument(
                    "addHiddenLayer: dtype '" + std::string(dtype_name(dtype)) + "' does not match "
                    "the scalar type of the function pointers given ('" + std::string(dtype_name(dtype_of<T>::value)) + "')");

            // A layer whose activation has no registered derivative would dereference a
            // null function pointer on its first backward pass.
            if (core::functions::activations::fetchAssociatedDerivative(activation) == nullptr)
                throw std::invalid_argument(
                    "addHiddenLayer: activation has no derivative registered in core/activations.h, "
                    "so the layer cannot be trained through");

            LayerBase* layer = new HiddenLayer<T>(
                width, inputSize,
                dtype,
                activation,
                init, loss
            );
            delete hiddenLayers[layerIndex];   // replacing a filled slot must not leak
            hiddenLayers[layerIndex] = layer;
        }

        // Convenience overload: the dtype is implied by the scalar type T.
        template <typename T>
        void addHiddenLayer(
            int layerIndex,
            int width, int inputSize,
            core::functions::activations::ActivationFunction<T> activation,
            core::functions::initializers::initFunction<T> init,
            core::functions::loss::LossFunction<T> loss
        )
        {
            addHiddenLayer(layerIndex, width, inputSize, dtype_of<T>::value, activation, init, loss);
        }

        template <typename T>
        void addOutputLayer(
            int layerIndex,
            int width, int inputSize,
            DType dtype,
            core::functions::activations::ActivationFunction<T> activation,
            core::functions::initializers::initFunction<T> init,
            core::functions::loss::LossFunction<T> loss
        )
        {
            /**
             * @brief Adds an output layer at the specified index with the given configuration.
             * @param layerIndex The index at which to add the output layer.
             * @param width The number of neurons in the output layer.
             * @param inputSize The size of the input vector to the output layer.
             * @param dtype The scalar type the layer computes in; must match the type of the
             *              activation/initializer/loss function pointers.
             * @param activation The activation function for the output layer.
             * @param init The weight initializer function.
             * @param loss The loss function for the output layer.
             */
            if (layerIndex < 0 || layerIndex >= static_cast<int>(outputLayers.size()))
                throw std::out_of_range(
                    "addOutputLayer: layerIndex " + std::to_string(layerIndex) +
                    " is outside the " + std::to_string(outputLayers.size()) + " output slots");

            if (dtype != dtype_of<T>::value)
                throw std::invalid_argument(
                    "addOutputLayer: dtype '" + std::string(dtype_name(dtype)) + "' does not match "
                    "the scalar type of the function pointers given ('" + std::string(dtype_name(dtype_of<T>::value)) + "')");

            if (core::functions::activations::fetchAssociatedDerivative(activation) == nullptr)
                throw std::invalid_argument(
                    "addOutputLayer: activation has no derivative registered in core/activations.h, "
                    "so the layer cannot be trained through");
            if (core::functions::loss::fetchAssociatedDerivative(loss) == nullptr)
                throw std::invalid_argument(
                    "addOutputLayer: loss has no derivative registered in core/loss.h, "
                    "so the layer cannot be trained against");

            LayerBase* layer = new OutputLayer<T>(
                width, inputSize,
                dtype,
                activation,
                init, loss
            );
            delete outputLayers[layerIndex];   // replacing a filled slot must not leak
            outputLayers[layerIndex] = layer;
        }

        // Convenience overload: the dtype is implied by the scalar type T.
        template <typename T>
        void addOutputLayer(
            int layerIndex,
            int width, int inputSize,
            core::functions::activations::ActivationFunction<T> activation,
            core::functions::initializers::initFunction<T> init,
            core::functions::loss::LossFunction<T> loss
        )
        {
            addOutputLayer(layerIndex, width, inputSize, dtype_of<T>::value, activation, init, loss);
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

        void forwardPass(const IO* inputs)
        {
            /**
             * @brief Performs a forward pass through the network, calculating the outputs for all layers.
             * @param inputs The input vector to the network, in the IO dtype. (So per epoch basis the inputs will be passed)
             *  The inputs are passed through pointers so we need the inputs to survive the stack during the whole process.
             *  Each layer converts the incoming activations into its own dtype at the boundary.
             */
            const void* __restrict__ currentInput = inputs;
            DType currentDtype = ioDType;
            // Just sequationally call the calculateOutputs function for each layer
            // We cannot parallelize this easily because each layer depends on the output of the previous layer.
            // Thus no SIMD either
            for (auto& layer : hiddenLayers)
            {
                layer->calculateOutputs(currentInput, currentDtype);
                currentInput = layer->activatedOutputs();
                currentDtype = layer->dtype;
            }
            for (auto& layer : outputLayers)
            {
                layer->calculateOutputs(currentInput, currentDtype);
                currentInput = layer->activatedOutputs();
                currentDtype = layer->dtype;
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

        void accumulateGradients(std::vector<IO>& targetOutputs)
        {
            /**
             * @brief Accumulates gradients for all layers in the network based on the provided target outputs.
             * @param targetOutputs The target output vectors for the current batch, in the IO dtype.
             *
             * We have moved from the older versions as we now batch and accumalate our gradients.
             * Why?
             * Accumulating gradients over a batch helps in stabilizing the training process and can lead
             * to better convergence compared to updating weights after every single sample. [form my research on this topic]
             */
            std::vector<IO>& currentTarget = targetOutputs;

            for (int i = this->numOutputLayers - 1; i >= 0; i--)
            {
                // Go through the output layers, this will in 99% cases just be a single layer
                LayerBase* currentLayer = this->outputLayers[i];
                currentLayer->computeErrorSignal(currentTarget.data(), ioDType);
                currentLayer->computeGradients();
            }

            // Transfer the results from the output layer into aliases
            // These aliases will be used to propagate the error signals backward through the hidden layers.
            int nextLayerWidth = this->outputLayers[0]->layerWidth;
            const void* nextLayerDeltas = this->outputLayers[0]->deltas();
            const void* nextLayerWeights = this->outputLayers[0]->weights();
            DType nextDtype = this->outputLayers[0]->dtype;

            // Go through the hidden layers in reverse order to propagate the error signals backward.
            for (int i = this->numHiddenLayers - 1; i >= 0; i--)
            {
                // Current hidden layer to process
                LayerBase* currentLayer = this->hiddenLayers[i];
                currentLayer->computeErrorSignal(nextLayerWidth, nextLayerDeltas, nextLayerWeights, nextDtype);
                currentLayer->computeGradients();

                // Store for next iterations
                nextLayerWidth = currentLayer->layerWidth;
                nextLayerDeltas = currentLayer->deltas();
                nextLayerWeights = currentLayer->weights();
                nextDtype = currentLayer->dtype;
            }
        }

        IO computeLoss(const std::vector<IO>& targetOutputs) const
        {
            /**
             * @brief Computes the loss of the network for the given target outputs.
             * @param targetOutputs The expected output values for the current input, in the IO dtype.
             * @return The computed loss value.
             */
            IO totalLoss = static_cast<IO>(0);
            size_t targetOffset = 0;

            // Compute the total loss by summing the loss from each output layer. [Usually just one output layer]
            for (const auto& layer : outputLayers)
            {
                totalLoss += static_cast<IO>(layer->computeLoss(
                    targetOutputs.data() + targetOffset,
                    ioDType
                ));
                targetOffset += static_cast<size_t>(layer->layerWidth);
            }
            // Return the total loss over all output layers. [In most cases, this is just the loss of the single output layer]
            return totalLoss;
        }

        IO trainBatch(const size_t* __restrict__ sampleOrder, size_t startSample, int batchSize,
                      std::vector<std::vector<IO>>& inputs,
                      std::vector<std::vector<IO>>& targetOutputs)
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

            IO batchLoss = static_cast<IO>(0);
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
            return batchLoss / static_cast<IO>(batchSize);
        }

        IO trainBatch(size_t startSample, int batchSize,
                      std::vector<std::vector<IO>>& inputs,
                      std::vector<std::vector<IO>>& targetOutputs)
        {
            /**
             * @brief Trains on a batch of samples in their natural order.
             * Convenience overload for callers that do not shuffle.
             */
            return trainBatch(nullptr, startSample, batchSize, inputs, targetOutputs);
        }

        void train(std::vector<std::vector<IO>>& inputs,
                   std::vector<std::vector<IO>>& targetOutputs,
                   int numEpochs, IO* __restrict__ lossBuffer, int bufferSize,
                   bool verbose = false, bool shuffle = false,
                   std::optional<unsigned> seed = std::nullopt, int logEvery = 1)
        {
            /**
             * @brief Trains the network for a number of epochs over the whole dataset.
             * @param inputs One row per sample.
             * @param targetOutputs One row per sample, aligned with inputs.
             * @param numEpochs How many passes over the dataset to make.
             * @param lossBuffer Receives the mean loss of each epoch; must hold numEpochs IO scalars.
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
                IO epochLoss = static_cast<IO>(0);

                // Process all samples in batches.
                size_t sample = 0;
                while (sample < numSamples)
                {
                    const int thisBatch = static_cast<int>(
                        std::min<size_t>(static_cast<size_t>(batchSize), numSamples - sample)
                    );
                    const IO batchLoss = trainBatch(order, sample, thisBatch, inputs, targetOutputs);

                    epochLoss += batchLoss * static_cast<IO>(thisBatch);
                    sample += static_cast<size_t>(thisBatch);
                }

                // Store and display the average loss for the current epoch.
                const IO meanLoss = epochLoss / static_cast<IO>(numSamples);
                lossBuffer[epoch] = meanLoss;
                if (verbose && (epoch % logEvery == 0 || epoch == numEpochs - 1))
                    std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs
                              << ", Loss: " << meanLoss << std::endl;
            }
        }
    };
}
