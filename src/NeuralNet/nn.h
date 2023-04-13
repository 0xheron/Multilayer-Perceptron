#pragma once

#include <Eigen/Dense>

// std
#include <vector>

class NeuralNet
{
private:
    // The number of layers in the neural network
    size_t layers;

    // The number of output neurons in each layer
    std::vector<size_t> neurons_per_layer;

    // All of the weights in the neural network
    std::vector<Eigen::MatrixX2d> weights;

    // All of the biases in the neural network
    std::vector<Eigen::VectorXd> biases;
public:
    // Creates a neural network with the specified number of neurons per layer and the specified number of layers
    // Initalizes weights and biases to random values
    NeuralNet(const std::vector<size_t>& neurons_per_layer);

    // Feeds the input through the neural network and returns the output
    Eigen::VectorXd feed_forward(const Eigen::VectorXd& input);
};