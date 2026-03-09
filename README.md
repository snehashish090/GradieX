# NeuralCpp

A work-in-progress neural network engine written in modern C++.

This project is currently in the **early prototype stage**. The codebase already sketches core building blocks:

- `Neuron` class
- `Layer` class
- `NeuralNetwork` class
- Multiple activation functions (sigmoid, tanh, ReLU, linear, leaky ReLU, softmax helper)

## Current Status

What exists right now:

- Basic class structure for neurons, layers, and network organization
- Forward-pass intent in `NeuralNetwork::forward_pass()`
- Initial activation function implementations

What is still in progress:

- Compile/runtime correctness for all classes
- Weight initialization strategy
- Matrix/vector shape handling across layers
- Training pipeline (loss calculation, backpropagation, optimizer)
- Dataset loading and batching

## Project Goals

- Build a lightweight neural network engine from scratch in C++
- Understand and control the internals of forward and backward propagation
- Keep dependencies minimal and code educational/readable

## Planned Features

- Dense (fully connected) layers
- Configurable activations per layer
- Loss functions (MSE, cross-entropy)
- Backpropagation and gradient descent
- Model save/load
- Training loop with metrics
- Small example tasks (XOR, regression, classification)

## Build and Run (Windows)

If you have GCC/MinGW installed:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o neuralcpp
./neuralcpp
```

If you use MSVC (Developer Command Prompt):

```bat
cl /std:c++17 /EHsc main.cpp
main.exe
```

## Suggested Project Structure (Next Step)

```text
NeuralCpp/
  include/
    neuron.hpp
    layer.hpp
    network.hpp
    activations.hpp
    losses.hpp
  src/
    neuron.cpp
    layer.cpp
    network.cpp
    activations.cpp
    losses.cpp
  examples/
    xor.cpp
  tests/
  main.cpp
  README.md
```

## Development Notes

- The current implementation stores vectors directly inside each neuron.
- As the engine grows, consider separating math operations into utility modules.
- Add tests early for activation functions and forward pass to prevent regressions.

## Contributing

This repository is in active experimentation mode. Suggestions, refactors, and bug fixes are welcome.

## License

Add a license file (for example, MIT) before public distribution.
