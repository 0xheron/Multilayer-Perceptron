#include "NeuralNet/nn.h"
#include <iostream>
#include <fstream>
#include <vector>

// Read data from a file loaded from filepath into a string
std::vector<uint8_t> read_file(const std::string& filepath)
{
    std::ifstream file(filepath);
    std::vector<uint8_t> bytes;

    if (!file.is_open()) throw std::runtime_error("Cannot find file");

    file.seekg(0, std::ios::end);   
    bytes.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read((char*) bytes.data(), bytes.size());
    
    file.close();

    return bytes;
}

int main(void)
{
    std::cout << "Hello, Wordl!" << std::endl;

    // Super hacky dataset loading
    auto training_img = read_file("data/train-images-idx3-ubyte");
    auto training_lables = read_file("data/train-labels-idx1-ubyte");
    auto test_img = read_file("data/t10k-images-idx3-ubyte");
    auto test_lables = read_file("data/t10k-labels-idx1-ubyte");

    // Create a neural network with 784 input neurons, 512 hidden neurons, and 10 output neurons
    NeuralNet nn({ 784, 512, 10 });

    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> test_data;

    for (size_t i = 0; i < training_lables.size() - 8; i++)
    {
        Eigen::VectorXd input = Eigen::VectorXd(784);
        Eigen::VectorXd expected_output = Eigen::VectorXd(10);
        expected_output.setZero();
        expected_output[training_lables[i + 8]] = 1;

        for (size_t j = 0; j < 784; j++)
        {
            input[j] = training_img[16 + i * 784 + j] / 255.0;
        }

        training_data.push_back(std::pair(input, expected_output));
    }

    for (size_t i = 0; i < test_lables.size() - 8; i++)
    {
        Eigen::VectorXd input = Eigen::VectorXd(784);
        Eigen::VectorXd expected_output = Eigen::VectorXd(10);
        expected_output.setZero();
        expected_output[test_lables[i + 8]] = 1;

        for (size_t j = 0; j < 784; j++)
        {
            input[j] = test_img[16 + i * 784 + j] / 255.0;
        }

        test_data.push_back(std::pair(input, expected_output));
    }

    std::cout << "Training..." << std::endl;

    nn.train(training_data, 30, 64, 0.01, test_data);
}