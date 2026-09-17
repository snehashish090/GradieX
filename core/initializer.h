/*
    @file core/initializer.h
    @brief Provides various weight initialization functions for neural network layers, 
        including Xavier and He initializations.

    - The architecture follows the common pattern of providing initialization functions that 
        can be used for different types of layers in a neural network.
    - An example implmentation
        core::functions::initializers::initFunction __initFunction__;
        __initFunction__ = core::functions::initializers::zeros;
        float* initializedWeights = __initFunction__(vectorScratchPad, size, size2);
    - Supports both zero initialization and random initialization methods like Xavier and He.
    - Designed to be easily extendable for additional initialization methods in the future.
    - Namespaces are important for organizing code and avoiding name conflicts.
*/

#pragma once
#include <iostream>
#include <ostream>
#include <random>
#include <vector>
#include <functional>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <stdexcept>

/*
Note : we will not utilise compiler-specific vectorization pragmas like #pragma omp simd or prallel or gcc unroll
Why? : because these functions utilise std random generators which we cannot guarantee will be vectorized.
        Thus, we leave it up to the compiler.
*/
namespace core::functions::initializers
{
    inline std::mt19937& get_generator() {
        /*
            Returns a reference to the shared random number generator used by every
            initializer. Non-deterministically seeded unless setSeed() is called.
            @param None
            @return Reference to a static random number generator.
        */
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

    inline void setSeed(unsigned int seed) {
        /*
            Seeds the shared generator so that a run's weight initialisation is
            reproducible. Call before initializeAllWeightsAndBiases().
            @param seed The seed value.
            @return void
        */
        get_generator().seed(seed);
    }

    // Type alias for initialization functions that take a scratch pad and two size parameters 
    // and return a pointer to the initialized weights.
    using initFunction = float* (*)(float* __restrict__ vectorScratchPad, size_t size, size_t size2);

    inline float* zeros(float* __restrict__ vectorScratchPad, size_t size, size_t size2) {
        /*
            Zeros out the given scratch pad.
            @param vectorScratchPad Pointer to the scratch pad to be initialized.
            @param size The first dimension of the scratch pad.
            @param size2 The second dimension of the scratch pad.
            @return Pointer to the initialized scratch pad.
        */
        for (size_t i = 0; i < size * size2; ++i)
            vectorScratchPad[i] = 0.0f;
        return vectorScratchPad;
    }

    // Xavier Initialization: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    inline float* xavier(float* __restrict__ vectorScratchPad, 
                        size_t fanIn, size_t fanOut) {
        /*
            Xavier initialization for the given scratch pad.
            @param vectorScratchPad Pointer to the scratch pad to be initialized.
            @param fanIn The number of input units.
            @param fanOut The number of output units.
            @return Pointer to the initialized scratch pad.
        */
        // Calculate the correct standard deviation using the sum
        float std_dev = std::sqrt(2.0f / static_cast<float>(fanIn + fanOut));
        std::normal_distribution<float> dist(0.0f, std_dev);

        auto& gen = get_generator();
        size_t total_size = fanIn * fanOut;
        for (size_t i = 0; i < total_size; ++i) {
            vectorScratchPad[i] = dist(gen);
        }
        return vectorScratchPad;
    }

    inline float* he(float* __restrict__ vectorScratchPad,
                        size_t fanIn, size_t fanOut) {
        /*
            He initialization for the given scratch pad.  W ~ N(0, sqrt(2 / fan_in))
            @param vectorScratchPad Pointer to the scratch pad to be initialized.
            @param fanIn The number of input units.
            @param fanOut The number of output units.
            @return Pointer to the initialized scratch pad.
        */
        float std_dev = std::sqrt(2.0f / static_cast<float>(fanIn));
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        auto& gen = get_generator();

        size_t total_size = fanIn * fanOut;
        for (size_t i = 0; i < total_size; ++i) {
            vectorScratchPad[i] = dist(gen);
        }
        return vectorScratchPad;
    }
}