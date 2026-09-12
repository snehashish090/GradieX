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


namespace core::functions::initializers
{
    inline std::mt19937& get_generator() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

    using initFunction = double* (*)(double* __restrict__ vectorScratchPad, size_t size, size_t size2);

    inline double* zeros(double* __restrict__ vectorScratchPad, size_t size, size_t size2) {
        for (size_t i = 0; i < size * size2; ++i)
            vectorScratchPad[i] = 0.0;
        return vectorScratchPad;
    }

    // Xavier Initialization: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    inline double* xavier(double* __restrict__ vectorScratchPad, 
                        size_t fanIn, size_t fanOut) {

        // FanIn - The number of inputs 
        // FanOut - The number of outputs

        // Calculate the correct standard deviation using the sum
        double std_dev = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
        std::normal_distribution<double> dist(0.0, std_dev);
        
        auto& gen = get_generator();
        size_t total_size = fanIn * fanOut;

        for (size_t i = 0; i < total_size; ++i) {
            vectorScratchPad[i] = dist(gen);
        }
        return vectorScratchPad;
    }

    inline double* he(double* __restrict__ vectorScratchPad,
                        size_t fanIn, size_t fanOut) {
        // He Initialization: W ~ N(0, sqrt(2 / fan_in))
        double std_dev = std::sqrt(2.0 / static_cast<double>(fanIn));
        std::normal_distribution<double> dist(0.0, std_dev);
        
        auto& gen = get_generator();

        size_t total_size = fanIn * fanOut;
        for (size_t i = 0; i < total_size; ++i) {
            vectorScratchPad[i] = dist(gen);
        }
        return vectorScratchPad;
    }
}