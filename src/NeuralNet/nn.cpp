#include "nn.h"

#include <cstdlib>
#include <iostream>

NeuralNet::NeuralNet(const std::vector<size_t>& neurons_per_layer)
{
    this->layers = neurons_per_layer.size();
    this->neurons_per_layer = neurons_per_layer;
    this->weights = std::vector<Eigen::MatrixXd>(this->layers);
    this->biases = std::vector<Eigen::VectorXd>(this->layers);
    this->activation_function = relu;
    this->derivative_activation_function = derivative_relu;
    
    for (size_t i = 0; i < layers - 1; i++)
    {
        this->weights[i] = Eigen::MatrixXd::Random(neurons_per_layer[i + 1], neurons_per_layer[i]);
        this->biases[i] = Eigen::VectorXd::Random(neurons_per_layer[i + 1]);
    }
}

NeuralNet::NeuralNet(const std::vector<size_t>& neurons_per_layer, std::function<act_fun_type> activation_function, std::function<act_fun_type> derivative_activation_function)
{
    this->layers = neurons_per_layer.size();
    this->neurons_per_layer = neurons_per_layer;
    this->weights = std::vector<Eigen::MatrixXd>(this->layers);
    this->biases = std::vector<Eigen::VectorXd>(this->layers);
    this->activation_function = activation_function;
    this->derivative_activation_function = derivative_activation_function;
    
    for (size_t i = 0; i < layers - 1; i++)
    {
        this->weights[i] = Eigen::MatrixXd::Random(neurons_per_layer[i + 1], neurons_per_layer[i]);
        this->biases[i] = Eigen::VectorXd::Random(neurons_per_layer[i + 1]);
    }
}

Eigen::VectorXd NeuralNet::feed_forward(const Eigen::VectorXd& input)
{
    Eigen::VectorXd result = input;
    for (size_t i = 0; i < layers; i++)
    {
        result = activation_function((weights[i] * result) + biases[i]);
    }
    return result;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> NeuralNet::compute_gradient(const Eigen::VectorXd& input, const Eigen::VectorXd& expected_output)
{
    std::vector<Eigen::MatrixXd> nabla_w = std::vector<Eigen::MatrixXd>(this->layers - 1);
    std::vector<Eigen::VectorXd> nabla_b = std::vector<Eigen::VectorXd>(this->layers - 1);
    
    // Feed forward
    std::vector<Eigen::VectorXd> activations = std::vector<Eigen::VectorXd>(this->layers);
    std::vector<Eigen::VectorXd> zs = std::vector<Eigen::VectorXd>(this->layers);
    activations[0] = input;

    for (size_t i = 0; i < layers - 1; i++)
    {
        zs[i] = (weights[i] * activations[i]) + biases[i];
        activations[i + 1] = activation_function(zs[i]);
    }

    
    // Backpropagation
    Eigen::VectorXd delta_error = hadamard((activations[activations.size() - 1] - expected_output), (derivative_activation_function(zs[zs.size() - 2])));
    nabla_w[nabla_w.size() - 1] = hadamard(delta_error, activations[activations.size() - 2]);
    nabla_b[nabla_b.size() - 1] = delta_error;
    for (size_t i = 1; i < layers - 1; i++)
    {
        delta_error = hadamard(delta_error.transpose() * weights[weights.size() - i - 1], derivative_activation_function(zs[zs.size() - i - 2]));
        nabla_w[nabla_w.size() - i - 1] = hadamard(delta_error, activations[activations.size() - i - 1]);
        nabla_b[nabla_b.size() - i - 1] = delta_error;
    }
    
    return std::pair(nabla_w, nabla_b);
}

void NeuralNet::train(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data, size_t epochs, size_t mini_batch_size, double learning_rate, const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data)
{
    for (size_t i = 0; i < epochs; i++)
    {
        for (int j = training_data.size() - 1; j > 0; j--)
        {
            // Fisher-Yates shuffle
            size_t k = rand() % j;
            training_data[j] = training_data[k];
        }

        for (size_t j = 0; j < training_data.size() / mini_batch_size; j++)
        {
            std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> gradient;
            gradient.first = std::vector<Eigen::Xd>(this->layers);
            gradient.second = std::vector<Eigen::VectorXd>(this->layers);

            for (size_t i = 0; i < layers - 1; i++)
            {
                gradient.first[i] = Eigen::MatrixXd::Zero(neurons_per_layer[i + 1], neurons_per_layer[i]);
                gradient.second[i] = Eigen::VectorXd::Zero(neurons_per_layer[i + 1]);
            }

            // Iterate over data 
            for (size_t k = 0; k < mini_batch_size; k++)
            {
                auto current_batch = std::vector(training_data.begin() + j * mini_batch_size, training_data.begin() + (j + 1) * mini_batch_size);
                auto add_to_grad = compute_gradient(current_batch[k].first, current_batch[k].second);
                for (size_t x = 0; x < layers - 1; x++)
                {
                    gradient.first[x] += add_to_grad.first[x];
                    gradient.second[x] += add_to_grad.second[x];
                }
            }

            for (size_t k = 0; k < layers - 1; k++)
            {
                this->weights[k] -= learning_rate * (gradient.first[k] / mini_batch_size);
                this->biases[k] -= learning_rate * (gradient.second[k] / mini_batch_size);
            }
        }
        
        if (test_data.size() > 0)
        {
            std::cout << "Epoch " << i << ": " << evaluate(test_data) << " / " << test_data.size() << std::endl;
        }
        else
        {
            std::cout << "Epoch " << i << " complete" << std::endl;
        }
    }
}

size_t NeuralNet::evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data)
{
    size_t correct = 0;
    for (auto& data : test_data)
    {
        if (feed_forward(data.first) == data.second)
        {
            correct++;
        }
    }
    return correct;
}