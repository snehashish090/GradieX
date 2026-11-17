/**
    @file core/loss.h
    @brief This file contains various loss functions and their derivatives for neural networks.

    - The architecture is the same as activations.h. LossFunction becomes the data type alias 
        which : is a function pointer to a loss function.
    - The required unit can then store the loss function as a member variable like :
        core::functions::loss::LossFunction<float> __lossFunction__;
    - This makes it simple to swap loss functions as needed without having to write a 
        bunch of switch cases or if-else chains in every function.
        __lossFunction__ = core::functions::loss::mse<float>;
    - Every function and both aliases are templates on the scalar type T (float, double), so a
        layer can run its loss in whatever precision the layer itself was built with.
*/

#pragma once
#include "compiler.h"
#include <functional>
#include <cmath>
#include <algorithm>

/**
    Throughout this code, we use #pragma omp simd to instruct the compiler to vectorize the loops for better performance.

    SIMD - Single Instruction, Multiple Data.
    It is a hardware-level execution model that allows a single CPU core to execute a single instruction simultaneously across 
    multiple data points in a single clock cycle.

    It essentially vectorizes the computation, allowing multiple data points to be processed in parallel, 
    which can significantly improve performance for large-scale numerical operations.
*/

namespace core::functions::loss {

    /*
        The smallest probability the log-based losses will look at.

        For a float32 engine this resolves about 1.2e-7 near 1.0, so the
        1e-12 this used to use rounded away completely: `1.0f - 1e-12f` IS
        `1.0f`, the upper clamp never bit, and a sigmoid that saturated to 1.0
        produced log(0) = -inf and a 1/0 derivative. Multiplied by the
        activation derivative (0.0 once saturated) that is inf * 0 = NaN, which
        reaches every weight on the next update and never leaves.

        1e-7 is representable either side of 1.0 in both float32 and float64,
        so both ends of the clamp hold and a saturated output yields a
        large-but-finite gradient.
    */
    template <typename T>
    inline constexpr T kProbabilityEpsilon = static_cast<T>(1e-7);

    // Function pointer aliases so layers can store loss implementations directly.
    template <typename T>
    using LossFunction = T (*) (const T* __restrict__ actual,
        const T* __restrict__ predicted, size_t size);
    template <typename T>
    using LossFunctionDerivative = T* (*) (const T* __restrict__ actual,
        const T* __restrict__ predicted,
        T* __restrict__ vectorScratchPad, size_t size);

    template <typename T>
    inline T mse(const T* __restrict__ actual,
                 const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.
 
            @return loss which is a scalar value representative of the whoke network
        */
        T sum = static_cast<T>(0);
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            T diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / static_cast<T>(size);
    }
    template <typename T>
    inline T* mseDerivative(const T* __restrict__ actual,
                            const T* __restrict__ predicted,
                            T* __restrict__ vectorScratchPad, size_t size) {
        /***
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise MSE gradients.
        */
        T invN = static_cast<T>(1) / static_cast<T>(size);
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = static_cast<T>(2) * (predicted[i] - actual[i]) * invN;
        return vectorScratchPad;
    }

    template <typename T>
    inline T mae(const T* __restrict__ actual,
                 const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.

            @return mean absolute error over all elements.
        */
        T sum = static_cast<T>(0);
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i)
            sum += std::abs(predicted[i] - actual[i]);
        return sum / static_cast<T>(size);
    }
    template <typename T>
    inline T* maeDerivative(const T* __restrict__ actual,
                            const T* __restrict__ predicted,
                            T* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise MAE subgradients.
        */
        T invN = static_cast<T>(1) / static_cast<T>(size);
        for (size_t i = 0; i < size; ++i) {
            T diff = predicted[i] - actual[i];
            // Subgradient: undefined exactly at 0, we pick 0
            vectorScratchPad[i] = (diff > static_cast<T>(0)) ? invN : (diff < static_cast<T>(0)) ? -invN : static_cast<T>(0);
        }
        return vectorScratchPad;
    }

    template <typename T>
    inline T huber(const T* __restrict__ actual,
                   const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.

            @return mean Huber loss over all elements.
        */
        constexpr T delta = static_cast<T>(1);  // tweak to taste
        T sum = static_cast<T>(0);
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            T diff = std::abs(predicted[i] - actual[i]);
            sum += (diff <= delta) ? static_cast<T>(0.5) * diff * diff
                                   : delta * (diff - static_cast<T>(0.5) * delta);
        }
        return sum / static_cast<T>(size);
    }
    template <typename T>
    inline T* huberDerivative(const T* __restrict__ actual,
                              const T* __restrict__ predicted,
                              T* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise Huber gradients.
        */
        constexpr T delta = static_cast<T>(1);  // must match huber()
        T invN = static_cast<T>(1) / static_cast<T>(size);
        for (size_t i = 0; i < size; ++i) {
            T diff = predicted[i] - actual[i];
            vectorScratchPad[i] = (std::abs(diff) <= delta) ? diff * invN
                                 : delta * ((diff > static_cast<T>(0)) ? invN : -invN);
        }
        return vectorScratchPad;
    }


    template <typename T>
    inline T binaryCrossEntropy(const T* __restrict__ actual,
                                const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains binary labels in [0, 1].
            @param predicted contains predicted probabilities in [0, 1].
            @param size is the number of elements in the vectors.

            @return mean binary cross-entropy.
        */
        constexpr T eps = kProbabilityEpsilon<T>;
        T sum = static_cast<T>(0);
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            T p = std::clamp(predicted[i], eps, static_cast<T>(1) - eps);
            sum += actual[i] * std::log(p) + (static_cast<T>(1) - actual[i]) * std::log(static_cast<T>(1) - p);
        }
        return -sum / static_cast<T>(size);
    }
    template <typename T>
    inline T* binaryCrossEntropyDerivative(const T* __restrict__ actual,
                                           const T* __restrict__ predicted,
                                           T* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains binary labels in [0, 1].
            @param predicted contains predicted probabilities in [0, 1].
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing BCE gradients wrt predictions.
        */
        constexpr T eps = kProbabilityEpsilon<T>;
        T invN = static_cast<T>(1) / static_cast<T>(size);
        for (size_t i = 0; i < size; ++i) {
            T p = std::clamp(predicted[i], eps, static_cast<T>(1) - eps);
            vectorScratchPad[i] = (p - actual[i]) / (p * (static_cast<T>(1) - p)) * invN;
        }
        return vectorScratchPad;
    }

    template <typename T>
    inline T crossEntropy(const T* __restrict__ actual,
                          const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains class probabilities.
            @param size is the number of classes.

            @return cross-entropy value over the class vector.
        */
        constexpr T eps = kProbabilityEpsilon<T>;
        T sum = static_cast<T>(0);
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i)
            sum += actual[i] * std::log(std::max(predicted[i], eps));
        return -sum;
    }
    template <typename T>
    inline T* crossEntropyDerivative(const T* __restrict__ actual,
                                     const T* __restrict__ predicted,
                                     T* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains class probabilities.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of classes.

            @return vectorScratchPad containing CE gradients wrt predictions.
        */
        constexpr T eps = kProbabilityEpsilon<T>;
        #pragma omp simd 
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = -actual[i] / std::max(predicted[i], eps);
        return vectorScratchPad;
    }

    template <typename T>
    inline T softmaxCrossEntropy(const T* __restrict__ actual,
                                 const T* __restrict__ predicted, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains logits (not probabilities).
            @param size is the number of classes.

            @return softmax cross-entropy computed via log-sum-exp.
        */
        // Find max logit for numerical stability
        T maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // sum of exponents
        T sumExp = static_cast<T>(0);
        #pragma omp simd reduction(+:sumExp)
        for (size_t i = 0; i < size; ++i) {
            sumExp += std::exp(predicted[i] - maxLogit);
        }

        // cross-entropy loss directly using log-sum-exp
        T loss = static_cast<T>(0);
        T logSumExp = std::log(sumExp);
        #pragma omp simd reduction(+:loss)
        for (size_t i = 0; i < size; ++i) {
            loss += actual[i] * (predicted[i] - maxLogit - logSumExp);
        }
        return -loss;
    }

    template <typename T>
    inline T* softmaxCrossEntropyDerivative(const T* __restrict__ actual,
                                            const T* __restrict__ predicted,
                                            T* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains logits (not probabilities).
            @param vectorScratchPad is reused to hold exp(logit) values and output gradients.
            @param size is the number of classes.

            @return vectorScratchPad containing dL/dlogits = softmax(logits) - target.
        */
        // Find max logit
        T maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // Compute exponents and store them temporarily in scratchPad
        T sumExp = static_cast<T>(0);
        #pragma omp simd reduction(+:sumExp)
        for (size_t i = 0; i < size; ++i) {
            vectorScratchPad[i] = std::exp(predicted[i] - maxLogit);
            sumExp += vectorScratchPad[i];
        }

        // Calculate final gradient: softmax(predicted) - actual
        T invSumExp = static_cast<T>(1) / sumExp;
        // No reduction clause: this loop OVERWRITES vectorScratchPad, it does not
        // accumulate into it. The `reduction(+:vectorScratchPad[:size])` that used
        // to sit here gave each lane a private copy initialised to the identity (0)
        // and summed them at the end, which is simply the wrong answer -- and it
        // only looked harmless because nothing enabled the pragmas. Building with
        // -fopenmp-simd made it fail immediately.
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            T prob = vectorScratchPad[i] * invSumExp;
            vectorScratchPad[i] = prob - actual[i];
        }
        
        return vectorScratchPad;
    }

    template <typename T>
    inline LossFunctionDerivative<T> fetchAssociatedDerivative(LossFunction<T> lossFunction)
    {
        if (lossFunction == mse<T>)                 return mseDerivative<T>;
        if (lossFunction == crossEntropy<T>)        return crossEntropyDerivative<T>;
        if (lossFunction == softmaxCrossEntropy<T>) return softmaxCrossEntropyDerivative<T>;
        if (lossFunction == binaryCrossEntropy<T>)  return binaryCrossEntropyDerivative<T>;
        if (lossFunction == mae<T>)                 return maeDerivative<T>;
        if (lossFunction == huber<T>)               return huberDerivative<T>;
        return nullptr;

    }
}
