#include <iostream>
#include <vector>
#include <chrono>
#include "core/layer.h"
#include "core/activations.h"
#include "core/initializer.h"
#include "core/network.h"

int main()
{
    // Binary classification dataset: XOR or simple geometric check (e.g. x + y > 0.5)
    // Input size: 2 features, Output size: 1 (binary classification)
    
    std::vector<std::vector<double>> trainInputs = {
        {0.1, 0.2},
        {0.8, 0.9},
        {0.2, 0.1},
        {0.9, 0.7},
        {0.4, 0.5},
        {0.6, 0.4}
    };
    
    std::vector<std::vector<double>> trainTargets = {
        {0.0},
        {1.0},
        {0.0},
        {1.0},
        {0.0},
        {1.0}
    };

    // Configure Network: 2 inputs -> Hidden layer (4 neurons) -> Output layer (1 neuron)
    core::Network network(1, 1, 0.3);

    network.addHiddenLayer(
        0,
        2,  // width
        2,  // input size
        core::functions::activations::sigmoid,
        core::functions::activations::sigmoidDerivative,
        core::functions::initializers::xavier,
        core::functions::loss::mse,
        core::functions::loss::mseDerivative
    );

    network.addOutputLayer(
        0,
        1,  // width
        2,  // input size
        core::functions::activations::sigmoid,
        core::functions::activations::sigmoidDerivative,
        core::functions::initializers::xavier,
        core::functions::loss::mse,
        core::functions::loss::mseDerivative
    );

    network.initializeAllWeightsAndBiases();
    network.printAllLayers();
    
    std::vector<double> lossBuffer;
    lossBuffer.resize(10000);

    network.train(trainInputs, trainTargets, 10000, lossBuffer.data(), static_cast<int>(lossBuffer.size()));
    return 0;
}