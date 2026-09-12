#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <cmath>

namespace core::functions::activations 
{
    //We use double*__restrict__ to point to a vector input
    // Why? Using __restrict__ tells the compiler that the pointer is the only reference to that memory,
    // which helps with optimization and vectorization.
    using ActivationFunction = double*__restrict__ (*)(const double*__restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size);
    using ActivationFunctionDerivative = double*__restrict__ (*)(const double*__restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size);

    // The following function accepts a pointer to a scratchPad vector which is a 
    // singular vector created once and reused for efficiency.
    // It allows us to hold one specific space in memory for the output, 
    // reducing the need for repeated allocations.
    
    // Sigmoid activation function - maps input values to the range (0, 1)
    inline double*__restrict__ sigmoid(const double*__restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
            #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = 1.0 / (1.0 + std::exp(-input[i]));
        return vectorScratchPad;
    }
    inline double*__restrict__ sigmoidDerivative(const double*__restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double sigmoidValue = 1.0 / (1.0 + std::exp(-input[i]));
            vectorScratchPad[i] = sigmoidValue * (1.0 - sigmoidValue);
        }
        return vectorScratchPad;
    }

    // Fastest activation. Zero for negatives, linear for positives.
    inline double* __restrict__ relu(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0) ? input[i] : 0.0;
        return vectorScratchPad;
    }
    inline double* __restrict__ reluDerivative(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0) ? 1.0 : 0.0;
        return vectorScratchPad;
    }

    // Leaky ReLU activation function - allows a small, non-zero gradient when the input is negative
    inline double* __restrict__ leakyRelu(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0) ? input[i] : 0.01 * input[i];
        return vectorScratchPad;
    }
    inline double* __restrict__ leakyReluDerivative(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = (input[i] > 0.0) ? 1.0 : 0.01;
        return vectorScratchPad;
    }

    // The TanH function - Hyperbolic tangent activation function
    // Note: used most commonly for traditional RNNs and LSTMs
    inline double* __restrict__ tanh(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = std::tanh(input[i]);  // std::tanh is highly optimized
        return vectorScratchPad;
    }
    inline double* __restrict__ tanhDerivative(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double t = std::tanh(input[i]);
            vectorScratchPad[i] = 1.0 - t * t;
        }
        return vectorScratchPad;
    }

    // x * sigmoid(x). Smooth, non-monotonic. Used in deep vision & some NLP.
    inline double* __restrict__ swish(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double x = input[i];
            vectorScratchPad[i] = x / (1.0 + std::exp(-x));  // x * sigmoid
        }
        return vectorScratchPad;
    }

    inline double* __restrict__ swishDerivative(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double x = input[i];
            double s = 1.0 / (1.0 + std::exp(-x));  // sigmoid
            // derivative = s + x * s * (1 - s)
            vectorScratchPad[i] = s + x * s * (1.0 - s);
        }
        return vectorScratchPad;
    }

    // Fast approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    inline double* __restrict__ gelu(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double x = input[i];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            vectorScratchPad[i] = 0.5 * x * (1.0 + std::tanh(inner));
        }
        return vectorScratchPad;
    }

    inline double* __restrict__ geluDerivative(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        // We use constexpr to load the constants at compile time for efficiency.
        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;
        constexpr double du_coeff = 0.134145;  
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i) {
            double x = input[i];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            double tanh_inner = std::tanh(inner);
            double sech2 = 1.0 - tanh_inner * tanh_inner;  // derivative of tanh is sech^2
            double du_dx = sqrt_2_over_pi * (1.0 + du_coeff * x * x);  // derivative of inner
            // d/dx [0.5 * x * (1 + tanh(inner))] 
            // = 0.5*(1+tanh) + 0.5*x*sech2*du_dx
            vectorScratchPad[i] = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * du_dx;
        }
        return vectorScratchPad;
    }
    // Btw, Use this in combination with cross-entropy loss for classification tasks.
    // The derivative is a jacobian matrix, but when combined with cross-entropy loss, it simplifies nicely.
    inline double* __restrict__ softmax(const double* __restrict__ input, 
        double* __restrict__ vectorScratchPad, size_t size) {
        // Find max for numerical stability (prevents exp overflow)
        double maxVal = input[0];
        for (size_t i = 1; i < size; ++i)
            if (input[i] > maxVal) maxVal = input[i];
        // Compute exponentials and sum
        double sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            double e = std::exp(input[i] - maxVal);
            vectorScratchPad[i] = e;
            sum += e;
        }
        // Normalize
        double invSum = 1.0 / sum;
        #pragma GCC unroll 4 // Tells compiler to unroll the loop iterations to maximize performance.
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] *= invSum;
        return vectorScratchPad;
    }
}