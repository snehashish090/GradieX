//
// Created by Snehashish Laskar on 9/9/2026.
//

#include <functional>
#include <cmath>
#include <algorithm>


namespace core::functions::loss {

    // Loss: compares two arrays, returns a single scalar value
    using LossFunction = double (*) (const double* __restrict__ actual,
        const double* __restrict__ predicted, size_t size);

    // Derivative: writes the gradient w.r.t. `predicted` into the scratchPad
    using LossFunctionDerivative = double* (*) (const double* __restrict__ actual,
        const double* __restrict__ predicted,
        double* __restrict__ vectorScratchPad, size_t size);

    inline double mse(const double* __restrict__ actual,
                      const double* __restrict__ predicted, size_t size) {
        double sum = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / static_cast<double>(size);
    }
    inline double* mseDerivative(const double* __restrict__ actual,
                                 const double* __restrict__ predicted,
                                 double* __restrict__ vectorScratchPad, size_t size) {
        double invN = 1.0 / static_cast<double>(size);
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = 2.0 * (predicted[i] - actual[i]) * invN;
        return vectorScratchPad;
    }

    inline double mae(const double* __restrict__ actual,
                      const double* __restrict__ predicted, size_t size) {
        double sum = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i)
            sum += std::abs(predicted[i] - actual[i]);
        return sum / static_cast<double>(size);
    }
    inline double* maeDerivative(const double* __restrict__ actual,
                                 const double* __restrict__ predicted,
                                 double* __restrict__ vectorScratchPad, size_t size) {
        double invN = 1.0 / static_cast<double>(size);
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double diff = predicted[i] - actual[i];
            // Subgradient: undefined exactly at 0, we pick 0
            vectorScratchPad[i] = (diff > 0.0) ? invN : (diff < 0.0) ? -invN : 0.0;
        }
        return vectorScratchPad;
    }

    inline double huber(const double* __restrict__ actual,
                        const double* __restrict__ predicted, size_t size) {
        constexpr double delta = 1.0;  // tweak to taste
        double sum = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double diff = std::abs(predicted[i] - actual[i]);
            sum += (diff <= delta) ? 0.5 * diff * diff
                                   : delta * (diff - 0.5 * delta);
        }
        return sum / static_cast<double>(size);
    }
    inline double* huberDerivative(const double* __restrict__ actual,
                                   const double* __restrict__ predicted,
                                   double* __restrict__ vectorScratchPad, size_t size) {
        constexpr double delta = 1.0;  // must match huber()
        double invN = 1.0 / static_cast<double>(size);
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double diff = predicted[i] - actual[i];
            vectorScratchPad[i] = (std::abs(diff) <= delta) ? diff * invN
                                 : delta * ((diff > 0.0) ? invN : -invN);
        }
        return vectorScratchPad;
    }


    inline double binaryCrossEntropy(const double* __restrict__ actual,
                                     const double* __restrict__ predicted, size_t size) {
        constexpr double eps = 1e-12;
        double sum = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double p = std::clamp(predicted[i], eps, 1.0 - eps);
            sum += actual[i] * std::log(p) + (1.0 - actual[i]) * std::log(1.0 - p);
        }
        return -sum / static_cast<double>(size);
    }
    inline double* binaryCrossEntropyDerivative(const double* __restrict__ actual,
                                                const double* __restrict__ predicted,
                                                double* __restrict__ vectorScratchPad, size_t size) {
        constexpr double eps = 1e-12;
        double invN = 1.0 / static_cast<double>(size);
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double p = std::clamp(predicted[i], eps, 1.0 - eps);
            vectorScratchPad[i] = (p - actual[i]) / (p * (1.0 - p)) * invN;
        }
        return vectorScratchPad;
    }

    inline double crossEntropy(const double* __restrict__ actual,
                               const double* __restrict__ predicted, size_t size) {
        constexpr double eps = 1e-12;
        double sum = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i)
            sum += actual[i] * std::log(std::max(predicted[i], eps));
        return -sum;
    }
    inline double* crossEntropyDerivative(const double* __restrict__ actual,
                                          const double* __restrict__ predicted,
                                          double* __restrict__ vectorScratchPad, size_t size) {
        constexpr double eps = 1e-12;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i)
            vectorScratchPad[i] = -actual[i] / std::max(predicted[i], eps);
        return vectorScratchPad;
    }

    inline double softmaxCrossEntropy(const double* __restrict__ actual,
                                      const double* __restrict__ predicted, size_t size) {
        // Find max logit for numerical stability
        double maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // sum of exponents
        double sumExp = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            sumExp += std::exp(predicted[i] - maxLogit);
        }

        // cross-entropy loss directly using log-sum-exp
        double loss = 0.0;
        double logSumExp = std::log(sumExp);
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            loss += actual[i] * (predicted[i] - maxLogit - logSumExp);
        }
        return -loss;
    }

    inline double* softmaxCrossEntropyDerivative(const double* __restrict__ actual,
                                                 const double* __restrict__ predicted,
                                                 double* __restrict__ vectorScratchPad, size_t size) {
        // Find max logit
        double maxLogit = predicted[0];
        for (size_t i = 1; i < size; ++i) {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // Compute exponents and store them temporarily in scratchPad
        double sumExp = 0.0;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            vectorScratchPad[i] = std::exp(predicted[i] - maxLogit);
            sumExp += vectorScratchPad[i];
        }

        // Calculate final gradient: softmax(predicted) - actual
        double invSumExp = 1.0 / sumExp;
        #pragma GCC unroll 4
        for (size_t i = 0; i < size; ++i) {
            double prob = vectorScratchPad[i] * invSumExp;
            vectorScratchPad[i] = prob - actual[i];
        }
        
        return vectorScratchPad;
    }

}
