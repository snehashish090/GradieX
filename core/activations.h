/*
    @file core/activations.h
    @brief This file contains various activation functions and their derivatives for neural networks.

    - The architecture is simple. ActivationFunction becomes the data type which is a function pointer to an activation function.
    - The required unit can then store the activation function as a member variable like :
        core::functions::activations::ActivationFunction __activationFunction__;
    - This makes it simple to swap activation functions as needed without having to write a bunch of switch cases or if-else chains in every function.
        __activationFunction__ = core::functions::activations::relu;
    - Namespaces are important for organizing code and avoiding name conflicts.
*/  
#pragma once
#include <vector>
#include <functional>
#include <cstddef>
#include <cmath>

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
    //We use float*__restrict__ to point to a vector input
    // Why? Using __restrict__ tells the compiler that the pointer is the only reference to that memory,
    // which helps with optimization and vectorization.

    // Custom Data Type alias for function pointers
    using ActivationFunction = float*__restrict__ (*)(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size);
    using ActivationFunctionDerivative = float*__restrict__ (*)(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size);

    // The following function accepts a pointer to a scratchPad vector which is a 
    // singular vector created once and reused for efficiency.
    // It allows us to hold one specific space in memory for the output, 
    // reducing the need for repeated allocations.
    
    inline float*__restrict__ identity(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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

    inline float*__restrict__ identityDerivative(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            vectorScratchPad[i] = 1.0f;
        return vectorScratchPad;
    }
    
    // Sigmoid activation function - maps input values to the range (0, 1)
    inline float*__restrict__ sigmoid(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            vectorScratchPad[i] = 1.0f / (1.0f + std::expf(-input[i]));
        return vectorScratchPad;
    }

    inline float*__restrict__ sigmoidDerivative(const float*__restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            float sigmoidValue = 1.0f / (1.0f + std::expf(-input[i]));
            vectorScratchPad[i] = sigmoidValue * (1.0f - sigmoidValue);
        }
        return vectorScratchPad;
    }

    // Fastest activation. Zero for negatives, linear for positives.
    inline float* __restrict__ relu(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
        /*
            ReLU activation function sets all negative input values to zero and keeps positive values unchanged.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
        return vectorScratchPad;
    }

    inline float* __restrict__ reluDerivative(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
        /*
            ReLU activation function derivative returns 1 for positive input elements and 0 for non-positive elements.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0f) ? 1.0f : 0.0f;
        return vectorScratchPad;
    }

    // Leaky ReLU activation function - allows a small, non-zero gradient when the input is negative
    // Leaky ReLU activation function - allows a small, non-zero gradient when the input is negative
    inline float* __restrict__ leakyRelu(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            vectorScratchPad[i] = (input[i] > 0.0f) ? input[i] : 0.01f * input[i];
        return vectorScratchPad;
    }

    inline float* __restrict__ leakyReluDerivative(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            vectorScratchPad[i] = (input[i] > 0.0f) ? 1.0f : 0.01f;
        return vectorScratchPad;
    }

    // The TanH function - Hyperbolic tangent activation function
    // Note: used most commonly for traditional RNNs and LSTMs
    inline float* __restrict__ tanh(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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

    inline float* __restrict__ tanhDerivative(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            float t = std::tanh(input[i]);
            vectorScratchPad[i] = 1.0f - t * t;
        }
        return vectorScratchPad;
    }

    // x * sigmoid(x). Smooth, non-monotonic. Used in deep vision & some NLP.
    inline float* __restrict__ swish(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            float x = input[i];
            vectorScratchPad[i] = x / (1.0f + std::expf(-x));  // x * sigmoid
        }
        return vectorScratchPad;
    }

    inline float* __restrict__ swishDerivative(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
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
            float x = input[i];
            float s = 1.0f / (1.0f + std::expf(-x));  // sigmoid
            // derivative = s + x * s * (1 - s)
            vectorScratchPad[i] = s + x * s * (1.0f - s);
        }
        return vectorScratchPad;
    }

    // Fast approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.0f44715 * x^3) ))
    inline float* __restrict__ gelu(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {

        /*
            Fast approximation of the GELU (Gaussian Error Linear Unit) activation function.
            GELU is commonly used in transformer models and is defined as 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */

        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            vectorScratchPad[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
        return vectorScratchPad;
    }

    inline float* __restrict__ geluDerivative(const float* __restrict__ input, 
        float* __restrict__ vectorScratchPad, size_t size) {
        /*
            Fast approximation of the derivative of the GELU (Gaussian Error Linear Unit) activation function.
            The derivative is computed based on the approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )).

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        // We use constexpr to load the constants at compile time for efficiency.
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        constexpr float du_coeff = 0.134145f;  
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_inner = std::tanh(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;  // derivative of tanh is sech^2
            float du_dx = sqrt_2_over_pi * (1.0f + du_coeff * x * x);  // derivative of inner
            vectorScratchPad[i] = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * du_dx;
        }
        return vectorScratchPad;
    }

    // Btw, Use this in combination with cross-entropy loss for classification tasks.
    // The derivative is a jacobian matrix, but when combined with cross-entropy loss, it simplifies nicely.
    inline float* __restrict__ softmax(const float* __restrict__ input, 
        /**
            Softmax activation function computes the normalized exponential of each input element.
            It is commonly used in the output layer of classification models to produce probability distributions.

            @param input Pointer to the input vector.
            @param vectorScratchPad Pointer to the scratch pad vector for storing the output.
            @param size Number of elements in the input vector.
            @return Pointer to the output vector stored in the scratch pad.
        */
        float* __restrict__ vectorScratchPad, size_t size) {
        // Find max for numerical stability (prevents exp overflow)
        float maxVal = input[0];
        for (size_t i = 1; i < size; ++i)
            if (input[i] > maxVal) maxVal = input[i];
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float e = std::expf(input[i] - maxVal);
            vectorScratchPad[i] = e;
            sum += e;
        }
        float invSum = 1.0f / sum;
        #pragma omp simd
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] *= invSum;
        return vectorScratchPad;
    }

    inline ActivationFunctionDerivative fetchAssociatedDerivative(ActivationFunction activation) {
        /**
            Fetches the derivative function associated with the given activation function.

            @param activation The activation function for which the derivative is needed.
            @return The corresponding derivative function, or nullptr if no match is found.
        */
        // This function returns the corresponding derivative function for a given activation function.
        if (activation == swish) return swishDerivative;
        if (activation == relu) return reluDerivative;
        if (activation == leakyRelu) return leakyReluDerivative;
        if (activation == tanh) return tanhDerivative;
        if (activation == sigmoid) return sigmoidDerivative;
        if (activation == gelu) return geluDerivative;
        if (activation == identity) return identityDerivative;
        // softmax is deliberately absent: its derivative is a Jacobian, and the
        // usable form is fused with categorical cross-entropy (dL/dz = p - y),
        // which this engine does not implement yet.
        // Add more activation functions as needed
        return nullptr; // Return nullptr if no matching derivative is found
    }
}