/**
    @file core/loss.h
    @brief This file contains various loss functions and their derivatives for neural networks.

    - The architecture is the same as activations.h. LossFunction becomes the data type alias 
        which : is a function pointer to a loss function.
    - The required unit can then store the loss function as a member variable like :
        core::functions::loss::LossFunction __lossFunction__;
    - This makes it simple to swap loss functions as needed without having to write a 
        bunch of switch cases or if-else chains in every function.
        __lossFunction__ = core::functions::loss::mse;
*/

#pragma once
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

        This engine is float32, which resolves about 1.2e-7 near 1.0, so the
        1e-12 this used to use rounded away completely: `1.0f - 1e-12f` IS
        `1.0f`, the upper clamp never bit, and a sigmoid that saturated to 1.0
        produced log(0) = -inf and a 1/0 derivative. Multiplied by the
        activation derivative (0.0 once saturated) that is inf * 0 = NaN, which
        reaches every weight on the next update and never leaves.

        1e-7f is representable either side of 1.0, so both ends of the clamp
        hold and a saturated output yields a large-but-finite gradient.
    */
    constexpr float kProbabilityEpsilon = 1e-7f;

    // Function pointer aliases so layers can store loss implementations directly.
    using LossFunction = float (*) (const float* __restrict__ actual,
        const float* __restrict__ predicted, size_t size);
    using LossFunctionDerivative = float* (*) (const float* __restrict__ actual,
        const float* __restrict__ predicted,
        float* __restrict__ vectorScratchPad, size_t size);
    
    inline float mse(const float* __restrict__ actual,
                      const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.
 
            @return loss which is a scalar value representative of the whoke network
        */
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            float diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / static_cast<float>(size);
    }
    inline float* mseDerivative(const float* __restrict__ actual,
                                 const float* __restrict__ predicted,
                                 float* __restrict__ vectorScratchPad, size_t size) {
        /***
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise MSE gradients.
        */
        float invN = 1.0f / static_cast<float>(size);
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = 2.0f * (predicted[i] - actual[i]) * invN;
        return vectorScratchPad;
    }

    inline float mae(const float* __restrict__ actual,
                      const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.

            @return mean absolute error over all elements.
        */
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i)
            sum += std::abs(predicted[i] - actual[i]);
        return sum / static_cast<float>(size);
    }
    inline float* maeDerivative(const float* __restrict__ actual,
                                 const float* __restrict__ predicted,
                                 float* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise MAE subgradients.
        */
        float invN = 1.0f / static_cast<float>(size);
        for (size_t i = 0; i < size; ++i) {
            float diff = predicted[i] - actual[i];
            // Subgradient: undefined exactly at 0, we pick 0
            vectorScratchPad[i] = (diff > 0.0f) ? invN : (diff < 0.0f) ? -invN : 0.0f;
        }
        return vectorScratchPad;
    }

    inline float huber(const float* __restrict__ actual,
                        const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param size is the number of elements in the vectors.

            @return mean Huber loss over all elements.
        */
        constexpr float delta = 1.0f;  // tweak to taste
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(predicted[i] - actual[i]);
            // 0.5f, not 0.5: a double literal promotes this whole expression to
            // double in an engine that is float everywhere else, which both costs
            // precision-free work and stops the loop vectorising (clang reports
            // "loop not vectorized" for exactly this under -fopenmp-simd).
            sum += (diff <= delta) ? 0.5f * diff * diff
                                   : delta * (diff - 0.5f * delta);
        }
        return sum / static_cast<float>(size);
    }
    inline float* huberDerivative(const float* __restrict__ actual,
                                   const float* __restrict__ predicted,
                                   float* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains a vector of the ground truth values.
            @param predicted contains a vector of the predicted values.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing element-wise Huber gradients.
        */
        constexpr float delta = 1.0f;  // must match huber()
        float invN = 1.0f / static_cast<float>(size);
        for (size_t i = 0; i < size; ++i) {
            float diff = predicted[i] - actual[i];
            vectorScratchPad[i] = (std::abs(diff) <= delta) ? diff * invN
                                 : delta * ((diff > 0.0f) ? invN : -invN);
        }
        return vectorScratchPad;
    }


    inline float binaryCrossEntropy(const float* __restrict__ actual,
                                     const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains binary labels in [0, 1].
            @param predicted contains predicted probabilities in [0, 1].
            @param size is the number of elements in the vectors.

            @return mean binary cross-entropy.
        */
        constexpr float eps = kProbabilityEpsilon;
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            float p = std::clamp(predicted[i], eps, 1.0f - eps);
            sum += actual[i] * std::log(p) + (1.0f - actual[i]) * std::log(1.0f - p);
        }
        return -sum / static_cast<float>(size);
    }
    inline float* binaryCrossEntropyDerivative(const float* __restrict__ actual,
                                                const float* __restrict__ predicted,
                                                float* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains binary labels in [0, 1].
            @param predicted contains predicted probabilities in [0, 1].
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of elements in the vectors.

            @return vectorScratchPad containing BCE gradients wrt predictions.
        */
        constexpr float eps = kProbabilityEpsilon;
        float invN = 1.0f / static_cast<float>(size);
        for (size_t i = 0; i < size; ++i) {
            float p = std::clamp(predicted[i], eps, 1.0f - eps);
            vectorScratchPad[i] = (p - actual[i]) / (p * (1.0f - p)) * invN;
        }
        return vectorScratchPad;
    }

    inline float crossEntropy(const float* __restrict__ actual,
                               const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains class probabilities.
            @param size is the number of classes.

            @return cross-entropy value over the class vector.
        */
        constexpr float eps = kProbabilityEpsilon;
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; ++i)
            sum += actual[i] * std::log(std::max(predicted[i], eps));
        return -sum;
    }
    inline float* crossEntropyDerivative(const float* __restrict__ actual,
                                          const float* __restrict__ predicted,
                                          float* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains class probabilities.
            @param vectorScratchPad is the output buffer for dL/dy.
            @param size is the number of classes.

            @return vectorScratchPad containing CE gradients wrt predictions.
        */
        constexpr float eps = kProbabilityEpsilon;
        #pragma omp simd 
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = -actual[i] / std::max(predicted[i], eps);
        return vectorScratchPad;
    }

    inline float softmaxCrossEntropy(const float* __restrict__ actual,
                                      const float* __restrict__ predicted, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains logits (not probabilities).
            @param size is the number of classes.

            @return softmax cross-entropy computed via log-sum-exp.
        */
        // Find max logit for numerical stability
        float maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // sum of exponents
        float sumExp = 0.0f;
        #pragma omp simd reduction(+:sumExp)
        for (size_t i = 0; i < size; ++i) {
            sumExp += std::exp(predicted[i] - maxLogit);
        }

        // cross-entropy loss directly using log-sum-exp
        float loss = 0.0f;
        float logSumExp = std::log(sumExp);
        #pragma omp simd reduction(+:loss)
        for (size_t i = 0; i < size; ++i) {
            loss += actual[i] * (predicted[i] - maxLogit - logSumExp);
        }
        return -loss;
    }

    inline float* softmaxCrossEntropyDerivative(const float* __restrict__ actual,
                                                 const float* __restrict__ predicted,
                                                 float* __restrict__ vectorScratchPad, size_t size) {
        /**
            @param actual contains class targets (typically one-hot or soft labels).
            @param predicted contains logits (not probabilities).
            @param vectorScratchPad is reused to hold exp(logit) values and output gradients.
            @param size is the number of classes.

            @return vectorScratchPad containing dL/dlogits = softmax(logits) - target.
        */
        // Find max logit
        float maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // Compute exponents and store them temporarily in scratchPad
        float sumExp = 0.0f;
        #pragma omp simd reduction(+:sumExp)
        for (size_t i = 0; i < size; ++i) {
            vectorScratchPad[i] = std::exp(predicted[i] - maxLogit);
            sumExp += vectorScratchPad[i];
        }

        // Calculate final gradient: softmax(predicted) - actual
        float invSumExp = 1.0f / sumExp;
        // No reduction clause: this loop OVERWRITES vectorScratchPad, it does not
        // accumulate into it. The `reduction(+:vectorScratchPad[:size])` that used
        // to sit here gave each lane a private copy initialised to the identity (0)
        // and summed them at the end, which is simply the wrong answer -- and it
        // only looked harmless because nothing enabled the pragmas. Building with
        // -fopenmp-simd made it fail immediately.
        #pragma omp simd
        for (size_t i = 0; i < size; ++i) {
            float prob = vectorScratchPad[i] * invSumExp;
            vectorScratchPad[i] = prob - actual[i];
        }
        
        return vectorScratchPad;
    }

    inline LossFunctionDerivative fetchAssociatedDerivative(LossFunction lossFunction)
    {
        if (lossFunction == mse)                 return mseDerivative;
        if (lossFunction == crossEntropy)        return crossEntropyDerivative;
        if (lossFunction == softmaxCrossEntropy) return softmaxCrossEntropyDerivative;
        if (lossFunction == binaryCrossEntropy)  return binaryCrossEntropyDerivative;
        if (lossFunction == mae)                 return maeDerivative;
        if (lossFunction == huber)               return huberDerivative;
        return nullptr;

    }
}
