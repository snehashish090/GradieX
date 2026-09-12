#pragma once
#include "layer.h"
#include <stdexcept>

namespace core
{
    class Network
    {
        public:
        int numHiddenLayers;
        int numOutputLayers=1;
        double learningRate = 0.01;

        std::vector<HiddenLayer*> hiddenLayers;
        std::vector<OutputLayer*> outputLayers;

        Network() : numHiddenLayers(0), numOutputLayers(1), learningRate(0.01) {}
        Network(int numHiddenLayers, int numOutputLayers, double learningRate) 
            : numHiddenLayers(numHiddenLayers), numOutputLayers(numOutputLayers), 
            learningRate(learningRate) {
                hiddenLayers.resize(numHiddenLayers);
                outputLayers.resize(numOutputLayers);
            }
        
        void addHiddenLayer(
            int layerIndex,
            int width, int inputSize, 
            core::functions::activations::ActivationFunction activation,
            core::functions::activations::ActivationFunctionDerivative activationDerivative,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss,
            core::functions::loss::LossFunctionDerivative lossDerivative
        )
        {
            HiddenLayer* layer = new HiddenLayer(
                width, inputSize, 
                activation, activationDerivative,
                init, loss, lossDerivative
            );
            hiddenLayers[layerIndex] = layer;
        }

        void addOutputLayer(
            int layerIndex,
            int width, int inputSize, 
            core::functions::activations::ActivationFunction activation,
            core::functions::activations::ActivationFunctionDerivative activationDerivative,
            core::functions::initializers::initFunction init,
            core::functions::loss::LossFunction loss,
            core::functions::loss::LossFunctionDerivative lossDerivative
        )
        {
            OutputLayer* layer = new OutputLayer(
                width, inputSize, 
                activation, activationDerivative,
                init, loss, lossDerivative
            );
            outputLayers[layerIndex] = layer;
        }

        void initializeAllWeightsAndBiases()
        {
            for (auto& layer : hiddenLayers)
                layer->initializeWeightsAndBiases();
            for (auto& layer : outputLayers)
                layer->initializeWeightsAndBiases();
        }

        void printAllLayers()
        {
            for (auto& layer : hiddenLayers)
                layer->print();
            for (auto& layer : outputLayers)
                layer->print();
        }

        ~Network()
        {
            for (auto& layer : hiddenLayers)
                delete layer;
            for (auto& layer : outputLayers)
                delete layer;
        }

        void forwardPass(const double* inputs)
        {
            const double* currentInput = inputs;
            for (auto& layer : hiddenLayers)
            {
                layer->calculateOutputs(currentInput);
                currentInput = layer->neuronOutputsActivated.data();
            }
            for (auto& layer : outputLayers)
            {
                layer->calculateOutputs(currentInput);
                currentInput = layer->neuronOutputsActivated.data();
            }
        }

        void backwardPass(std::vector<double>& targetOutputs)
        {
            // Perform backward calculations on output layers first, then hidden layers
            std::vector<double>& currentTarget = targetOutputs;
            
            for (int i = this->numOutputLayers - 1; i >= 0; i--)
            {
                OutputLayer* currentLayer = this->outputLayers[i];
                currentLayer->computeErrorSignal(currentTarget.data());
                currentLayer->computeGradients();
            }
            int nextLayerWidth = this->outputLayers[0]->layerWidth; // Assume that there is an output layer present
            const double* nextLayerDeltas = this->outputLayers[0]->neuronDeltas.data();
            const double* nextLayerWeights = this->outputLayers[0]->neuronWeights.data();

            for (int i = this->numHiddenLayers - 1; i >= 0; i--)
            {
                HiddenLayer* currentLayer = this->hiddenLayers[i];
                currentLayer->computeErrorSignal(nextLayerWidth, nextLayerDeltas, nextLayerWeights);
                currentLayer->computeGradients();
         
                nextLayerWidth = currentLayer->layerWidth;
                nextLayerDeltas = currentLayer->neuronDeltas.data();
                nextLayerWeights = currentLayer->neuronWeights.data();
            }

            for (auto& layer : hiddenLayers)
                layer->updateWeightsAndBiases(this->learningRate);
            for (auto& layer : outputLayers)
                layer->updateWeightsAndBiases(this->learningRate);
        }

        void runEpoch(std::vector<double>& inputs, std::vector<double>&targetOutputs)
        {
            this->forwardPass(inputs.data());
            this->backwardPass(targetOutputs);
        }

        double computeLoss(const std::vector<double>& targetOutputs) const
        {
            double totalLoss = 0.0;
            int targetOffset = 0;
            for (auto& layer : outputLayers)
            {
                for (int n = 0; n < layer->layerWidth; n++)
                {
                    double diff = layer->neuronOutputsActivated[n] - targetOutputs[targetOffset + n];
                    totalLoss += diff * diff;
                }
                targetOffset += layer->layerWidth;
            }
            return totalLoss;
        }

        void train(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& targetOutputs, int numEpochs, 
            double* __restrict__  lossBuffer, int bufferSize)
        {
            if (inputs.size() != targetOutputs.size())
                throw std::invalid_argument("train: inputs and targetOutputs must hold the same number of samples");
            if (inputs.empty())
                return;

            const size_t numSamples = inputs.size();
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                double epochLoss = 0.0;
                for (size_t sample = 0; sample < numSamples; ++sample)
                {
                    runEpoch(inputs[sample], targetOutputs[sample]);
                    epochLoss += computeLoss(targetOutputs[sample]);
                }
                // Assumes that loss buffer has at least 'numEpochs' elements
                // Just to be safe, we resize the vector before we start appending

                if (lossBuffer==nullptr || bufferSize < numEpochs)
                    throw std::invalid_argument("train: lossBuffer must have at least 'numEpochs' elements and not be null");

                lossBuffer[epoch] = epochLoss / static_cast<double>(numSamples); // Capture losses
                std::cout << "Epoch " << epoch << ", Loss: "
                    << epochLoss / static_cast<double>(numSamples) << std::endl;
            }
        }
    };
}