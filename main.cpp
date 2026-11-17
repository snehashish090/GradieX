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
    
    std::vector<std::vector<float>> trainInputs = {
        {0.1, 0.2},
        {0.8, 0.9},
        {0.2, 0.1},
        {0.9, 0.7},
        {0.4, 0.5},
        {0.6, 0.4}
    };
    
    std::vector<std::vector<float>> trainTargets = {
        {0.0},
        {1.0},
        {0.0},
        {1.0},
        {0.0},
        {1.0}
    };

    // Configure Network: 2 inputs -> Hidden layer (4 neurons) -> Output layer (1 neuron)
    // The default Network is Network<float> (float32 in, float32 out). Every layer
    // also declares its own dtype as a constructor argument; here the hidden layer
    // runs in float64 while the rest of the network stays float32, and the engine
    // converts at the layer boundaries.
    core::Network network(1, 1, 0.3f);

    network.addHiddenLayer(
        0,
        2,  // width
        2,  // input size
        core::DType::float64,  // this hidden layer computes in double
        core::functions::activations::sigmoid<double>,
        core::functions::initializers::xavier<double>,
        core::functions::loss::mse<double>
    );

    network.addOutputLayer(
        0,
        1,  // width
        2,  // input size
        core::DType::float32,
        core::functions::activations::sigmoid<float>,
        core::functions::initializers::xavier<float>,
        core::functions::loss::mse<float>
    );

    network.initializeAllWeightsAndBiases();
    
    std::vector<float> lossBuffer;
    lossBuffer.resize(10000);

    network.train(trainInputs, trainTargets, 10000, lossBuffer.data(), static_cast<int>(lossBuffer.size()));
    return 0;
}
