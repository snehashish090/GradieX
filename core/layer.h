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

    v0.0.4 layering (per-layer data types):
        - Every layer's scalar type is a template parameter: Layer<float>, Layer<double>,
          HiddenLayer<float>, OutputLayer<double>, ... Each layer's buffers, activation,
          loss and initializer all live in that layer's own T.
        - The layer constructor takes the dtype as a *runtime* argument as well
          (`DType dtype`, see core/dtype.h) and validates that it matches T, so a layer
          can be described -- and checked -- by a value, not just by a template argument.
        - LayerBase is the type-erased interface a Network walks. Because two adjacent
          layers may disagree on T, every boundary crossing (forward input, backward
          deltas/weights, target) carries the source dtype at run time, and each layer
          converts incoming buffers into its own T once per call into a reused scratch
          buffer. A network whose layers all share one dtype does not pay for this:
          same-dtype conversion is a plain vectorisable copy.

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
#include <stdexcept>
#include <string>
#include "dtype.h"
#include "activations.h"
#include "initializer.h"
#include "loss.h"

namespace core
{
    // ---------------------------------------------------------------------------
    // LayerBase: the type-erased view of a layer.
    //
    // A Network stores LayerBase pointers so that adjacent layers can live in
    // different scalar types. The geometry and the dtype are visible here; the
    // typed buffers live in the Layer<T> template below. Inputs to the virtuals
    // that cross a layer boundary are raw pointers tagged with the dtype of the
    // layer that produced them.
    // ---------------------------------------------------------------------------
    class LayerBase
    {
        public:

        int layerWidth = 0; // Number of neurons in the layer
        int inputSize = 0;  // Number of inputs each neuron receives
        DType dtype = DType::float32; // The scalar type this layer computes in

        virtual ~LayerBase() = default;

        virtual void initializeWeightsAndBiases() = 0;
        virtual void calculateOutputs(const void* input, DType inputDtype) = 0;
        virtual void zeroGradients() = 0;
        virtual void scaleGradients(float scale) = 0;
        virtual void computeGradients() = 0;
        virtual void updateWeightsAndBiases(float learningRate) = 0;

        // The output-layer role (only OutputLayer<T> overrides this overload):
        virtual void computeErrorSignal(const void* /*target*/, DType /*targetDtype*/)
        {
            throw std::logic_error("computeErrorSignal(target, ...) is only implemented by output layers");
        }
        // The hidden-layer role (only HiddenLayer<T> overrides this overload):
        virtual void computeErrorSignal(int /*nextLayerWidth*/, const void* /*nextLayerDeltas*/,
                                        const void* /*nextLayerWeights*/, DType /*nextDtype*/)
        {
            throw std::logic_error("computeErrorSignal(next, ...) is only implemented by hidden layers");
        }
        // Loss is only defined on output layers; reported as double across the
        // type-erasure boundary so a double layer's loss keeps its precision.
        virtual double computeLoss(const void* /*target*/, DType /*targetDtype*/) const
        {
            throw std::logic_error("computeLoss() is only implemented by output layers");
        }

        // Raw views of this layer's buffers, in THIS layer's dtype. The Network
        // uses these to hand one layer's activations/deltas/weights to the next.
        virtual const void* activatedOutputs() const = 0;
        virtual const void* deltas() const = 0;
        virtual const void* weights() const = 0;

        // float32 / float64 copies of the parameter buffers. This is the bridge
        // the Python bindings (and tools) use to move weights/biases across the
        // boundary; values are converted between the requested type and the
        // layer's T as needed. A float64 Network reads float64 arrays here even
        // though the layers themselves may compute in float32.
        virtual void copyWeightsFrom(const float* src) = 0;
        virtual void copyWeightsFrom(const double* src) = 0;
        virtual void copyWeightsTo(float* dst) const = 0;
        virtual void copyWeightsTo(double* dst) const = 0;
        virtual void copyBiasesFrom(const float* src) = 0;
        virtual void copyBiasesFrom(const double* src) = 0;
        virtual void copyBiasesTo(float* dst) const = 0;
        virtual void copyBiasesTo(double* dst) const = 0;
        virtual void copyActivatedOutputsTo(float* dst) const = 0;
        virtual void copyActivatedOutputsTo(double* dst) const = 0;
    };

    // ---------------------------------------------------------------------------
    // Layer<T>: the concrete, typed layer.
    //
    // Everything the pre-template Layer did, but on T. The layer is constructed
    // with the dtype as an explicit argument; it must agree with T, so "the
    // layer is float64" is both a compile-time and a runtime fact.
    // ---------------------------------------------------------------------------
    template <typename T>
    class Layer : public LayerBase
    {
        /**
            Layer class represents a single layer in a neural network.

            Code Design : 
                - pointers to vector data come in <type>* __restrict__ for better performance and to enable compiler optimizations.
                - When writing to vector data in for loops, we establish __restrict__ pointer first 
                    eg : vector[i] = data WRONG DO NOT DO THIS 
                    
                    Correct way:
                    T* __restrict__ vec = vector.data();
                    vec[i] = data;

                    Why?  Using __restrict__ informs the compiler that the pointer is the only reference to that memory. 
        */
        // Class Members
        public:

        using ActivationFn = core::functions::activations::ActivationFunction<T>;
        using ActivationDerivativeFn = core::functions::activations::ActivationFunctionDerivative<T>;
        using InitFn = core::functions::initializers::initFunction<T>;
        using LossFn = core::functions::loss::LossFunction<T>;
        using LossDerivativeFn = core::functions::loss::LossFunctionDerivative<T>;

        ActivationFn activationFunction = nullptr; // Activation function for the layer
        ActivationDerivativeFn activationFunctionDerivative = nullptr; // Activation function derivative for the layer
        InitFn initialiser = nullptr; // Initialiser for the layer weights
        LossFn lossFunction = nullptr; // Loss function for the layer
        LossDerivativeFn lossFunctionDerivative = nullptr; // Loss function derivative for the layer
        std::vector<T> layerInputs;              // Caches the last input (in T) fed to calculateOutputs
        std::vector<T> neuronOutputsUnactivated; // Stores the z values
        std::vector<T> neuronOutputsActivated;   // Stores the activated outputs (a values)
        std::vector<T> neuronDeltas;             // Stores the error signals (delta values)
        std::vector<T> neuronWeights;            // Stores the weights of the layer as a 1D Matrix
        // Why? a vector of vector causes heap allocations.
        std::vector<T> neuronBiases;             // Stores the biases of the layer
        std::vector<T> scratchPad1;              // Temporary storage for intermediate calculations
        std::vector<T> scratchPad2;
        std::vector<T> neuronWeightGradients;    // Stores the gradients of the neurons
        std::vector<T> neuronBiasGradients;      // Stores the gradients of the neuron biases

        Layer() noexcept = default;

        Layer(int width, 
            int inputSize,
            DType dtype,
            ActivationFn activation,
            InitFn init,
            LossFn loss)
        {
            /**
             * Constructs a layer with the specified width, input size, dtype, activation function,
             * initializer, and loss function.
             * @param width The width of the layer
             * @param inputSize The size of the input vector
             * @param dtype The scalar type the layer computes in; must match T
             * @param activation The activation function for the layer
             * @param init The initializer function for the layer's weights
             * @param loss The loss function associated with the layer
             */

            // The runtime dtype argument must agree with the compile-time scalar
            // type, otherwise "this layer is float64" would be a lie one level up.
            if (dtype != dtype_of<T>::value)
                throw std::invalid_argument(
                    "Layer: dtype '" + std::string(dtype_name(dtype)) + "' does not match the "
                    "layer's scalar type '" + std::string(dtype_name(dtype_of<T>::value)) + "'");

            this->layerWidth = width;
            this->inputSize = inputSize;
            this->dtype = dtype;
            this->activationFunction = activation;
            // The derivatved is DERIVED from the activation function (lmao :D)
            this->activationFunctionDerivative = core::functions::activations::fetchAssociatedDerivative(activation);
            this->lossFunctionDerivative = core::functions::loss::fetchAssociatedDerivative(loss);
            this->initialiser = init;
            this->lossFunction = loss;
            
            // Reserve memory for all the vectors 
            this->layerInputs.resize(static_cast<size_t>(inputSize));
            this->neuronOutputsUnactivated.resize(static_cast<size_t>(width));
            this->neuronOutputsActivated.resize(static_cast<size_t>(width));
            this->neuronDeltas.resize(static_cast<size_t>(width));
            this->neuronWeights.resize(static_cast<size_t>(width) * static_cast<size_t>(inputSize));
            this->neuronBiases.resize(static_cast<size_t>(width));
            this->neuronWeightGradients.resize(static_cast<size_t>(width) * static_cast<size_t>(inputSize));
            this->neuronBiasGradients.resize(static_cast<size_t>(width));

            this->scratchPad1.resize(static_cast<size_t>(width)); // Allocate enough buffer
            this->scratchPad2.resize(static_cast<size_t>(width)); // Allocate enough buffer
        }

        // Convenience overload: the dtype is implied by the scalar type T.
        Layer(int width,
            int inputSize,
            ActivationFn activation,
            InitFn init,
            LossFn loss)
            : Layer(width, inputSize, dtype_of<T>::value, activation, init, loss)
        {
        }

        void initializeWeightsAndBiases() override
        {
            /**
             * Initializes the weights and biases of the layer.
             *
             * @param neuronWeights Pointer to the neuron's weights
             * @param inputSize The size of the input vector
             * @param layerWidth The width of the layer
             */
            
            T* __restrict__ neuronWeights = this->neuronWeights.data();
            T* __restrict__ neuronBiases = this->neuronBiases.data();

            // Initialize the weights and biases using the provided initialiser
            this->initialiser(
                neuronWeights, // The neuron's weights
                static_cast<size_t>(this->inputSize),
                static_cast<size_t>(this->layerWidth)
            );
            // Set the biases to zero initially {that is the general convention}
            for (int i = 0; i < this->layerWidth; ++i)
                neuronBiases[i] = static_cast<T>(0);
        }

        void calculateOutputs(const void* input, DType inputDtype) override
        {
            /**
             * Calculates the outputs of the layer given the input vector.
             * Vector Equation :  z^{[l]} = W^{[l]} * a^{[l-1]} + b^{[l]}
             * where 
             * z^{[l]} represents the unactivated outputs of the current layer
             * W^{[l]} represents the weight matrix, a^{[l-1]} represents the activated outputs of the previous layer
             * b^{[l]} represents the biases of the current layer.
             * @param input Pointer to the input vector, in `inputDtype` (the previous layer's dtype)
             * @param inputDtype The dtype of the input vector
             * @return void
             */

            T* __restrict__ layerInputs = this->layerInputs.data();
            T* __restrict__ neuronOutputsUnactivated = this->neuronOutputsUnactivated.data();
            T* __restrict__ neuronOutputsActivated = this->neuronOutputsActivated.data();
            const T* __restrict__ neuronWeights = this->neuronWeights.data();
            const T* __restrict__ neuronBiases = this->neuronBiases.data();

            const int width     = this->layerWidth; // Load ahead of time
            const int inputSize = this->inputSize;

            // Cache the inputs passed into the layer's member variable, converting
            // them into this layer's dtype when the previous layer differs.
            if (inputDtype == dtype_of<T>::value)
            {
                const T* __restrict__ src = static_cast<const T*>(input);
                #pragma omp simd
                for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                    layerInputs[inputIdx] = src[inputIdx];
            }
            else
            {
                convertFrom(static_cast<const void*>(input), inputDtype,
                            layerInputs, static_cast<size_t>(inputSize));
            }
            
            for (int neuron = 0; neuron < width; ++neuron)
            {
                const T* __restrict__ w = neuronWeights + neuron * inputSize;
                T sum = static_cast<T>(0);
                // Vectorise the dot product computation using OpenMP SIMD directive
                #pragma omp simd reduction(+:sum)
                for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                    sum += w[inputIdx] * layerInputs[inputIdx];
                neuronOutputsUnactivated[neuron] = sum + neuronBiases[neuron];
            }
            // Activate the output.
            this->activationFunction(neuronOutputsUnactivated,
                                    neuronOutputsActivated,
                                    static_cast<size_t>(width));
        }

        // Convenience overload for callers that already hold T buffers: the input
        // dtype is implied. (Direct calls with the wrong type simply do not compile.)
        void calculateOutputs(const T* input)
        {
            calculateOutputs(static_cast<const void*>(input), dtype_of<T>::value);
        }

        void zeroGradients() override
        {
            /**
             * Zeros out the gradients of the layer's weights and biases.
             * @return void
             * This is typically done at the beginning of a training step to ensure that gradients from previous steps do not accumulate.
             */
            // Set our local pointers to the vectors for easier access and potential SIMD optimization.
            T* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            T* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            const int totalWeights = this->layerWidth * this->inputSize;
            const int width        = this->layerWidth;
            // I think its pretty simple, just set all gradients & biases to zero.
            #pragma omp simd
            for (int i = 0; i < totalWeights; ++i)
                neuronWeightGradients[i] = static_cast<T>(0);

            #pragma omp simd
            for (int i = 0; i < width; ++i)
                neuronBiasGradients[i] = static_cast<T>(0);
        }

        void scaleGradients(float scale) override
        {
            /**
             * Scales the gradients of the layer's weights and biases by the given factor.
             * @param scale The factor by which to scale the gradients.
             * @return void
             * 
             * Why would we want to scale gradients? 
             * This is typically done to adjust the learning rate dynamically or to implement gradient clipping.
             */
            T* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            T* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            const int totalWeights = this->layerWidth * this->inputSize;
            const int width        = this->layerWidth;
            const T scaleT = static_cast<T>(scale);

            // Again, we use SIMD to efficiently scale all gradients in parallel.
            // And we simply multiply vector elements by the scale factor.
            #pragma omp simd
            for (int i = 0; i < totalWeights; ++i)
                neuronWeightGradients[i] *= scaleT;

            #pragma omp simd
            for (int i = 0; i < width; ++i)
                neuronBiasGradients[i] *= scaleT;
        }

        void computeGradients() override
        {
            /**
             * @brief Computes the gradients of the layer's weights and biases based on the current neuron deltas and inputs.
             * We will typically call this during the backpropagation step of training.
             * The computed gradients are stored internally and later used to update the weights and biases.
             * @return void
             * This function does not take any parameters as it operates on the internal state of the layer.
             */
            // Set vector pointers 
            const T* __restrict__ savedInputs = this->layerInputs.data();
            const T* __restrict__ neuronDeltas = this->neuronDeltas.data();
            T* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            T* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
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
                const T delta = neuronDeltas[neuron];
                T* __restrict__ dW = neuronWeightGradients + neuron * inputSize;

                #pragma omp simd
                for (int inputIdx = 0; inputIdx < inputSize; ++inputIdx)
                    dW[inputIdx] += delta * savedInputs[inputIdx];

                neuronBiasGradients[neuron] += delta;
            }
        }

        void updateWeightsAndBiases(float learningRate) override
        {
            /**
             * @brief Updates the weights and biases of the layer using the computed gradients.
             * @param learningRate The learning rate to scale the gradient updates.
             * @return void
             * This function updates the internal weights and biases of the layer based on the previously computed gradients.
             */
            const T* __restrict__ neuronWeightGradients = this->neuronWeightGradients.data();
            const T* __restrict__ neuronBiasGradients   = this->neuronBiasGradients.data();
            T* __restrict__ neuronWeights = this->neuronWeights.data();
            T* __restrict__ neuronBiases  = this->neuronBiases.data();
            const int width = this->layerWidth;
            const int inputSize = this->inputSize;
            const T lr = static_cast<T>(learningRate);

            // Again, we apply SIMD only to the inner loop over inputs for efficient vectorized updates.
            for (int neuron = 0; neuron < width; ++neuron)
            {
                #pragma omp simd
                for (int input = 0; input < inputSize; ++input)
                    neuronWeights[neuron * inputSize + input] -= 
                    lr * neuronWeightGradients[neuron * inputSize + input];
                neuronBiases[neuron] -= lr * neuronBiasGradients[neuron];
            }
        }

        // ------------------------------------------------------- LayerBase views

        const void* activatedOutputs() const override
        {
            return this->neuronOutputsActivated.data();
        }

        const void* deltas() const override
        {
            return this->neuronDeltas.data();
        }

        const void* weights() const override
        {
            return this->neuronWeights.data();
        }

        void copyWeightsFrom(const float* src) override
        {
            convert(src, this->neuronWeights.data(),
                    static_cast<size_t>(this->layerWidth) * static_cast<size_t>(this->inputSize));
        }

        void copyWeightsFrom(const double* src) override
        {
            convert(src, this->neuronWeights.data(),
                    static_cast<size_t>(this->layerWidth) * static_cast<size_t>(this->inputSize));
        }

        void copyWeightsTo(float* dst) const override
        {
            convert(this->neuronWeights.data(), dst,
                    static_cast<size_t>(this->layerWidth) * static_cast<size_t>(this->inputSize));
        }

        void copyWeightsTo(double* dst) const override
        {
            convert(this->neuronWeights.data(), dst,
                    static_cast<size_t>(this->layerWidth) * static_cast<size_t>(this->inputSize));
        }

        void copyBiasesFrom(const float* src) override
        {
            convert(src, this->neuronBiases.data(), static_cast<size_t>(this->layerWidth));
        }

        void copyBiasesFrom(const double* src) override
        {
            convert(src, this->neuronBiases.data(), static_cast<size_t>(this->layerWidth));
        }

        void copyBiasesTo(float* dst) const override
        {
            convert(this->neuronBiases.data(), dst, static_cast<size_t>(this->layerWidth));
        }

        void copyBiasesTo(double* dst) const override
        {
            convert(this->neuronBiases.data(), dst, static_cast<size_t>(this->layerWidth));
        }

        void copyActivatedOutputsTo(float* dst) const override
        {
            convert(this->neuronOutputsActivated.data(), dst, static_cast<size_t>(this->layerWidth));
        }

        void copyActivatedOutputsTo(double* dst) const override
        {
            convert(this->neuronOutputsActivated.data(), dst, static_cast<size_t>(this->layerWidth));
        }
    };

    // ---------------------------------------------------------------------------
    // OutputLayer<T>: computes its error signal from the target directly.
    // ---------------------------------------------------------------------------
    template <typename T>
    class OutputLayer : public Layer<T>
    {
        public:
            using Layer<T>::Layer;
            // Keep the type-erased overloads visible next to the T-typed convenience
            // overloads, so the derived class does not hide them (and warn about it).
            using LayerBase::computeErrorSignal;
            using LayerBase::computeLoss;

            void computeErrorSignal(const void* target, DType targetDtype) override
            {
                /**
                 * @brief Computes the error signal (deltas) for the output layer based on the target output.
                 * @param target Pointer to the array of target output values, in `targetDtype`
                 * (the network's IO dtype, which need not match this layer's dtype).
                 * @param targetDtype The dtype of the target vector.
                 * @return void
                 * This function calculates the error signal for each neuron in the output layer by computing the derivative of 
                 * the loss function with respect to the neuron's activated output and multiplying it by the derivative of the 
                 * activation function.
                 */
                const int width = this->layerWidth;

                // Convert the target into this layer's dtype; the scratch buffer is
                // allocated once and reused on every subsequent call.
                this->targetScratch_.resize(static_cast<size_t>(width));
                convertFrom(target, targetDtype, this->targetScratch_.data(), static_cast<size_t>(width));

                T* __restrict__ neuronDeltas = this->neuronDeltas.data();
                // Compute the derivative of the loss function with respect to the activated outputs.
                T* lossDerivative = this->lossFunctionDerivative(
                    this->targetScratch_.data(),
                    this->neuronOutputsActivated.data(),
                    this->scratchPad1.data(),
                    static_cast<size_t>(width)
                );
                // Compute the derivative of the activation function with respect to the unactivated outputs.
                T* activationDerivedInputs = this->activationFunctionDerivative(
                    this->neuronOutputsUnactivated.data(),
                    this->scratchPad2.data(),
                    static_cast<size_t>(width)
                );
                #pragma omp simd 
                // Compute the error signal for each neuron in the output layer.
                // Which is the element-wise product of the loss derivative and the activation function derivative.
                for (int neuron = 0; neuron < width; ++neuron)
                    neuronDeltas[neuron] = lossDerivative[neuron] * activationDerivedInputs[neuron];
            }

            // Convenience overload: the target is already in this layer's dtype.
            void computeErrorSignal(const T* target)
            {
                computeErrorSignal(static_cast<const void*>(target), dtype_of<T>::value);
            }

            double computeLoss(const void* target, DType targetDtype) const override
            {
                /**
                 * @brief Computes the loss of this layer against the target.
                 * @return The loss, widened to double across the type-erasure boundary.
                 */
                const int width = this->layerWidth;
                this->targetScratch_.resize(static_cast<size_t>(width));
                convertFrom(target, targetDtype, this->targetScratch_.data(), static_cast<size_t>(width));
                return static_cast<double>(this->lossFunction(
                    this->targetScratch_.data(),
                    this->neuronOutputsActivated.data(),
                    static_cast<size_t>(width)
                ));
            }

            // Convenience overload: the target is already in this layer's dtype.
            double computeLoss(const T* target) const
            {
                return computeLoss(static_cast<const void*>(target), dtype_of<T>::value);
            }

        private:
            mutable std::vector<T> targetScratch_; // converted copy of the last target
    };
    
    // ---------------------------------------------------------------------------
    // HiddenLayer<T>: computes its error signal from the next layer's deltas and
    // weights, which arrive in the next layer's dtype and are converted here.
    // ---------------------------------------------------------------------------
    template <typename T>
    class HiddenLayer : public Layer<T>
    {
        /**
         * What is different? 
         * The HiddenLayer computes its error signal based on the deltas and weights of the next layer, 
         * rather than directly from the target output as in the OutputLayer.
         */
    public:
        using Layer<T>::Layer;
        // Keep the type-erased overloads visible next to the T-typed convenience
        // overload, so the derived class does not hide them (and warn about it).
        using LayerBase::computeErrorSignal;

        void computeErrorSignal(int nextLayerWidth, 
            const void* nextLayerDeltas, 
            const void* nextLayerWeights,
            DType nextDtype) override
        {
            /**
             * @brief Computes the error signal (deltas) for the hidden layer based on the next layer's deltas and weights.
             * @param nextLayerWidth The number of neurons in the next layer.
             * @param nextLayerDeltas Pointer to the array of deltas from the next layer, in `nextDtype`.
             * @param nextLayerWeights Pointer to the array of weights connecting this layer to the next layer, in `nextDtype`.
             * @param nextDtype The dtype of both next-layer buffers.
             * @return void
             * This function calculates the error signal for each neuron in the hidden layer by propagating the errors from 
             * the next layer backward through the weights and applying the derivative of the activation function.
             */
            const int width = this->layerWidth;
            const size_t nextWeightCount = static_cast<size_t>(nextLayerWidth) * static_cast<size_t>(width);

            // Convert the next layer's buffers into this layer's dtype. The scratch
            // buffers are allocated on the first call and reused from then on, so
            // the steady-state backward pass performs no heap allocations.
            this->nextDeltaScratch_.resize(static_cast<size_t>(nextLayerWidth));
            this->nextWeightScratch_.resize(nextWeightCount);
            convertFrom(nextLayerDeltas, nextDtype, this->nextDeltaScratch_.data(), static_cast<size_t>(nextLayerWidth));
            convertFrom(nextLayerWeights, nextDtype, this->nextWeightScratch_.data(), nextWeightCount);

            T* __restrict__ neuronDeltas = this->neuronDeltas.data();

            // 1. Compute derivatives for the whole layer in one go.
            T* activationDerivedInputs = this->activationFunctionDerivative(
                this->neuronOutputsUnactivated.data(),
                this->scratchPad2.data(),
                static_cast<size_t>(width)
            );

            #pragma omp simd
            // Empty before insertion of propagated errors from the next layer.
            for (int neuron = 0; neuron < width; ++neuron)
                neuronDeltas[neuron] = static_cast<T>(0);

            for (int nextNeuron = 0; nextNeuron < nextLayerWidth; ++nextNeuron)
            {
                // Get the delta from the l+1 (next) layer
                const T nextDelta = this->nextDeltaScratch_[static_cast<size_t>(nextNeuron)];
                const T* __restrict__ nextWeightRow = this->nextWeightScratch_.data() + static_cast<size_t>(nextNeuron) * static_cast<size_t>(width);
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

        // Convenience overload: both next-layer buffers are already in this layer's dtype.
        void computeErrorSignal(int nextLayerWidth, const T* nextLayerDeltas, const T* nextLayerWeights)
        {
            computeErrorSignal(nextLayerWidth,
                               static_cast<const void*>(nextLayerDeltas),
                               static_cast<const void*>(nextLayerWeights),
                               dtype_of<T>::value);
        }

    private:
        std::vector<T> nextDeltaScratch_; // converted copy of the next layer's deltas
        std::vector<T> nextWeightScratch_; // converted copy of the next layer's weights
    };
}
