#pragma once

#include "util/util.h"

#include <Eigen/Dense>

// std
#include <vector>
#include <functional>

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

    // The activation function used in the neural network
    std::function<act_fun_type> activation_function;

    // The derivative of the activation function used in the neural network
    std::function<act_fun_type> derivative_activation_function;

public:
    // Creates a neural network with the specified number of neurons per layer and the specified number of layers
    // Initalizes weights and biases to random values

    // Sets the activation function as relu
    NeuralNet(const std::vector<size_t>& neurons_per_layer);

    // Allows the user to specify the activation function and its derivative
    NeuralNet(const std::vector<size_t>& neurons_per_layer, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_function, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> derivative_activation_function);

    // Feeds the input through the neural network and returns the output
    Eigen::VectorXd feed_forward(const Eigen::VectorXd& input);

    std::pair<std::vector<Eigen::MatrixX2d>, std::vector<Eigen::VectorXd>> compute_gradient(const Eigen::VectorXd& input, const Eigen::VectorXd& expected_output);

    // Trains the neural network using the specified training data
    void train(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data, size_t epochs, size_t mini_batch_size, double learning_rate, const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>());

    size_t evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data);
};