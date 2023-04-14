#include <iostream>
#include <fstream>
#include <memory>
#include <array>
#include <vector>
#include <stdexcept>

#include "NeuralNet/nn.h"

// Read data from a file loaded from filepath into a vector of bytes
std::vector<uint8_t> read_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Load datasets
    auto training_images = std::make_unique<std::vector<uint8_t>>(read_file("data/train-images-idx3-ubyte"));
    auto training_labels = std::make_unique<std::vector<uint8_t>>(read_file("data/train-labels-idx1-ubyte"));
    auto test_images = std::make_unique<std::vector<uint8_t>>(read_file("data/t10k-images-idx3-ubyte"));
    auto test_labels = std::make_unique<std::vector<uint8_t>>(read_file("data/t10k-labels-idx1-ubyte"));

    // Create a neural network with 784 input neurons, 512 hidden neurons, and 10 output neurons
    std::array<int, 3> layer_sizes{ 784, 512, 10 };
    NeuralNet nn(layer_sizes);

    // Convert datasets into vector of pairs of inputs and expected outputs
    auto convert_dataset = [](const auto& images, const auto& labels) {
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> data;
        for (size_t i = 8; i < labels.size(); i++) {
            Eigen::VectorXd input(784);
            Eigen::VectorXd expected_output(10);
            expected_output.setZero();
            expected_output[labels[i]] = 1;

            for (size_t j = 0; j < 784; j++) {
                input[j] = images[i * 784 + j] / 255.0;
            }

            data.push_back({ input, expected_output });
        }
        return data;
    };

    auto training_data = convert_dataset(*training_images, *training_labels);
    auto test_data = convert_dataset(*test_images, *test_labels);

