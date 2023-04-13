#include "nn.h"

NeuralNet::NeuralNet(const std::vector<size_t>& neurons_per_layer)
{
    this->layers = neurons_per_layer.size();
    this->neurons_per_layer = neurons_per_layer;
    this->weights = std::vector<Eigen::MatrixX2d>(this->layers);
    this->biases = std::vector<Eigen::VectorXd>(this->layers);
    
    for (size_t i = 0; i < layers; i++)
    {
        this->weights[i] = Eigen::MatrixX2d::Random(neurons_per_layer[i], 2);
        this->biases[i] = Eigen::VectorXd::Random(neurons_per_layer[i]);
    }
}

NeuralNet::feed_forward(const Eigen::VectorXd& input)
{
    Eigen::VectorXd result = input;
    for (size_t i = 0; i < layers; i++)
    {
        result = (weights[i] * result) + biases[i];
    }
    return result;
}

NeuralNet::backpropagate()
{
    // Dont understand this ahhh
}