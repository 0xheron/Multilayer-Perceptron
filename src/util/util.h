#pragma once

#include <iostream>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>

// A type alias for the return value of activation functions
using act_fun_type = Eigen::VectorXd(const Eigen::VectorXd&);

// Calculate hadamard multiplication of two vectors (used to calculate delta error)
inline Eigen::VectorXd hadamard(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    Eigen::VectorXd result(a.size());
    for (int i = 0; i < a.size(); i++)
    {
        result[i] = b[i];
    }
    return result;
}

// A bunch of utitities for neural networks such as activation functions, loss functions, etc.
inline double single_relu(double x)
{
    return x > 0 ? x : 0;
}

inline Eigen::VectorXd relu(const Eigen::VectorXd& x)
{
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
    {
        result[i] = single_relu(x[i]);
    }
    return result;
}

inline double single_derivative_relu(double x)
{
    return x > 0 ? 1 : 0;
}

inline Eigen::VectorXd derivative_relu(const Eigen::VectorXd& x)
{
    Eigen::VectorXd result(x.size());
    for (int i = 0; i < x.size(); i++)
    {
        result[i] = single_derivative_relu(x[i]);
    }
    return result;
}