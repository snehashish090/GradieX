# GradieX

GradieX is a lightweight neural network engine written in modern C++ with a companion web front-end.

## Current Core Capabilities

- Dense feedforward neural network with configurable hidden/output activations
- End-to-end forward pass across multiple hidden layers
- Backward pass with gradient propagation and per-parameter updates
- Loss functions: mean squared error and binary cross entropy
- Training loop with epoch logging and inference API
- Built-in XOR training example in `main.cpp`

## Build and Run on Linux

Compile:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o gradiex-linux-x86_64
```

Run:

```bash
./gradiex-linux-x86_64
```

## Distribute Binary via the Web App

To make the Linux binary downloadable from the frontend app:

```bash
mkdir -p frontend/public/downloads
cp gradiex-linux-x86_64 frontend/public/downloads/gradiex-linux-x86_64
```

Then start/build the frontend:

```bash
cd frontend
npm install
npm run dev
```

The binary is served at:

- `/downloads/gradiex-linux-x86_64`

## Project Layout

```text
NebulaNeural/
  main.cpp
  README.md
  frontend/
    public/
    src/
```

## Next Steps

- Add mini-batch training and shuffled epochs
- Add model save/load for trained weights
- Add additional examples (regression and multi-class classification)
