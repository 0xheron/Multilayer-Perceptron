# Multilayer-Perceptron
A multilayer perceptron neural network made using c++ and Eigen for matrix math. No neural network libraries are used in this process. 

The feed forward network is fairly simple, as it is a matrix vector product of the weights of the current layer and the inputs of the current layer, added with the current layer biases
This can be done for each layer until the last layer is reached. That is how the network will run. 

Backpropagation. Come back when I understand this (actually, I am currently working on implementing this, I just didn't feel like writing all of the calculus right now) 

## To Use 
Currently only works with std::c++17 (Due to Eigen not compiling properly, might be a clang macros issue?)