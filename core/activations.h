/*
    @file core/activations.h
    @brief This file contains various activation functions and their derivatives for neural networks.

    - The architecture is simple. ActivationFunction becomes the data type which is a function pointer to an activation function.
    - The required unit can then store the activation function as a member variable like :
        core::functions::activations::ActivationFunction<float> __activationFunction__;
    - This makes it simple to swap activation functions as needed without having to write a bunch of switch cases or if-else chains in every function.
        __activationFunction__ = core::functions::activations::relu<float>;
    - Every function and both aliases are templates on the scalar type T (float, double), so a
        layer can run its activation in whatever precision the layer itself was built with.
    - Namespaces are important for organizing code and avoiding name conflicts.
*/  
#pragma once
#include <vector>
#include <functional>
#include <cstddef>
#include <cmath>
#include "compiler.h"

/*
    Throughout this code, we use #pragma omp simd to instruct the compiler to vectorize the loops for better performance.

    SIMD - Single Instruction, Multiple Data.
    It is a hardware-level execution model that allows a single CPU core to execute a single instruction simultaneously across 
    multiple data points in a single clock cycle.

    It essentially vectorizes the computation, allowing multiple data points to be processed in parallel, 
    which can significantly improve performance for large-scale numerical operations.
*/
namespace core::functions::activations 
{
    //We use T*__restrict__ to point to a vector input
    // Why? Using __restrict__ tells the compiler that the pointer is the only reference to that memory,
    // which helps with optimization and vectorization.

    // Custom Data Type alias for function pointers
    template <typename T>
    using ActivationFunction = T*__restrict__ (*)(const T*__restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size);
    template <typename T>
    using ActivationFunctionDerivative = T*__restrict__ (*)(const T*__restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size);

    // The following function accepts a pointer to a scratchPad vector which is a 
    // singular vector created once and reused for efficiency.
    // It allows us to hold one specific space in memory for the output, 
    // reducing the need for repeated allocations.
    
    template <typename T>
    inline T*__restrict__ identity(const T*__restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Identity activation function simply returns the input as output.
            No transformation is applied.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd // force vectorization
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = input[i];
        return vectorScratchPad;
    }

    template <typename T>
    inline T*__restrict__ identityDerivative(const T*__restrict__ /*input*/, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Identity activation function derivative returns 1 for all input elements.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = static_cast<T>(1);
        return vectorScratchPad;
    }
    
    // Sigmoid activation function - maps input values to the range (0, 1)
    template <typename T>
    inline T*__restrict__ sigmoid(const T*__restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Sigmoid activation function maps input values to the range (0, 1).
            It is defined as 1 / (1 + exp(-x)).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input[i]));
        return vectorScratchPad;
    }

    template <typename T>
    inline T*__restrict__ sigmoidDerivative(const T*__restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Sigmoid activation function derivative computes the derivative of the sigmoid function for each input element.
            The derivative is given by sigmoid(x) * (1 - sigmoid(x)).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            T sigmoidValue = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input[i]));
            vectorScratchPad[i] = sigmoidValue * (static_cast<T>(1) - sigmoidValue);
        }
        return vectorScratchPad;
    }

    // Fastest activation. Zero for negatives, linear for positives.
    template <typename T>
    inline T* __restrict__ relu(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            ReLU activation function sets all negative input values to zero and keeps positive values unchanged.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > static_cast<T>(0)) ? input[i] : static_cast<T>(0);
        return vectorScratchPad;
    }

    template <typename T>
    inline T* __restrict__ reluDerivative(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            ReLU activation function derivative returns 1 for positive input elements and 0 for non-positive elements.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
        return vectorScratchPad;
    }

    // Leaky ReLU activation function - allows a small, non-zero gradient when the input is negative
    template <typename T>
    inline T* __restrict__ leakyRelu(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Leaky ReLU activation function sets all negative input values to a small 
            fraction of the input value (0.01 * input) and keeps positive values unchanged.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > static_cast<T>(0)) ? input[i] : static_cast<T>(0.01) * input[i];
        return vectorScratchPad;
    }

    template <typename T>
    inline T* __restrict__ leakyReluDerivative(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Leaky ReLU activation function derivative computes the derivative of the Leaky ReLU function for each input element.
            The derivative is 1 for positive input values and 0.01 for negative input values.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0.01);
        return vectorScratchPad;
    }

    // The TanH function - Hyperbolic tangent activation function
    // Note: used most commonly for traditional RNNs and LSTMs
    template <typename T>
    inline T* __restrict__ tanh(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Hyperbolic tangent (TanH) activation function maps input values to the range [-1, 1].
            It is commonly used in traditional RNNs and LSTMs.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = std::tanh(input[i]);  // std::tanh is highly optimized
        return vectorScratchPad;
    }

    template <typename T>
    inline T* __restrict__ tanhDerivative(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Hyperbolic tangent (TanH) activation function derivative computes the derivative of the TanH function for each input element.
            The derivative is 1 - tanh^2(input).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T t = std::tanh(input[i]);
            vectorScratchPad[i] = static_cast<T>(1) - t * t;
        }
        return vectorScratchPad;
    }

    // x * sigmoid(x). Smooth, non-monotonic. Used in deep vision & some NLP.
    template <typename T>
    inline T* __restrict__ swish(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Swish activation function maps input values to x * sigmoid(x).
            It is smooth and non-monotonic, often used in deep vision and some NLP models.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T x = input[i];
            vectorScratchPad[i] = x / (static_cast<T>(1) + std::exp(-x));  // x * sigmoid
        }
        return vectorScratchPad;
    }

    template <typename T>
    inline T* __restrict__ swishDerivative(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Swish activation function derivative computes the derivative of the Swish function for each input element.
            The derivative is s + x * s * (1 - s), where s = sigmoid(x).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T x = input[i];
            T s = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));  // sigmoid
            // derivative = s + x * s * (1 - s)
            vectorScratchPad[i] = s + x * s * (static_cast<T>(1) - s);
        }
        return vectorScratchPad;
    }

    // Fast approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.0f44715 * x^3) ))
    template <typename T>
    inline T* __restrict__ gelu(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {

        /*
            Fast approximation of the GELU (Gaussian Error Linear Unit) activation function.
            GELU is commonly used in transformer models and is defined as 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */

        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);
        constexpr T coeff = static_cast<T>(0.044715);
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T x = input[i];
            T x_cubed = x * x * x;
            T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            vectorScratchPad[i] = static_cast<T>(0.5) * x * (static_cast<T>(1) + std::tanh(inner));
        }
        return vectorScratchPad;
    }

    template <typename T>
    inline T* __restrict__ geluDerivative(const T* __restrict__ input, 
        T* __restrict__ vectorScratchPad, size_t size) {
        /*
            Fast approximation of the derivative of the GELU (Gaussian Error Linear Unit) activation function.
            The derivative is computed based on the approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        // We use constexpr to load the constants at compile time for efficiency.
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);
        constexpr T coeff = static_cast<T>(0.044715);
        constexpr T du_coeff = static_cast<T>(0.134145);
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T x = input[i];
            T x_cubed = x * x * x;
            T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            T tanh_inner = std::tanh(inner);
            T sech2 = static_cast<T>(1) - tanh_inner * tanh_inner;  // derivative of tanh is sech^2
            T du_dx = sqrt_2_over_pi * (static_cast<T>(1) + du_coeff * x * x);  // derivative of inner
            vectorScratchPad[i] = static_cast<T>(0.5) * (static_cast<T>(1) + tanh_inner) + static_cast<T>(0.5) * x * sech2 * du_dx;
        }
        return vectorScratchPad;
    }

    // Btw, Use this in combination with cross-entropy loss for classification tasks.
    // The derivative is a jacobian matrix, but when combined with cross-entropy loss, it simplifies nicely.
    template <typename T>
    inline T* __restrict__ softmax(const T* __restrict__ input, 
        /**
            Softmax activation function computes the normalized exponential of each input element.
            It is commonly used in the output layer of classification models to produce probability distributions.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        T* __restrict__ vectorScratchPad, size_t size) {
        // Find max for numerical stability (prevents exp overflow)
        T maxVal = input[0];
        for (size_t i = 1; i < size; ++i)
            if (input[i] > maxVal) maxVal = input[i];
        T sum = static_cast<T>(0);
        for (size_t i = 0; i < size; ++i) {
            T e = std::exp(input[i] - maxVal);
            vectorScratchPad[i] = e;
            sum += e;
        }
        T invSum = static_cast<T>(1) / sum;
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] *= invSum;
        return vectorScratchPad;
    }

    template <typename T>
    inline ActivationFunctionDerivative<T> fetchAssociatedDerivative(ActivationFunction<T> activation) {
        /**
            Fetches the derivative function associated with the given activation function.

            @param activation The activation function for which the derivative is needed.
            @return The corresponding derivative function, or nullptr if no match is found.
        */
        // This function returns the corresponding derivative function for a given activation function.
        if (activation == swish<T>) return swishDerivative<T>;
        if (activation == relu<T>) return reluDerivative<T>;
        if (activation == leakyRelu<T>) return leakyReluDerivative<T>;
        if (activation == tanh<T>) return tanhDerivative<T>;
        if (activation == sigmoid<T>) return sigmoidDerivative<T>;
        if (activation == gelu<T>) return geluDerivative<T>;
        if (activation == identity<T>) return identityDerivative<T>;
        // softmax is deliberately absent: its derivative is a Jacobian, and the
        // usable form is fused with categorical cross-entropy (dL/dz = p - y),
        // which this engine does not implement yet.
        // Add more activation functions as needed
        return nullptr; // Return nullptr if no matching derivative is found
    }
}
