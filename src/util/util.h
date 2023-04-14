#pragma once

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>

// A type alias for the return value of activation functions
using act_fun_type = Eigen::VectorXd(const Eigen::VectorXd&);

// A bunch of utitities for neural networks such as activation functions, loss functions, etc.
inline double single_relu(double x)
{
    return x > 0 ? x : 0;
}

inline Eigen::VectorXd relu(const Eigen::VectorXd& x)
{
    Eigen::VectorXd result;
    for (int i = 0; i < result.size(); i++)
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
    Eigen::VectorXd result;
    for (int i = 0; i < result.size(); i++)
    {
        result[i] = single_derivative_relu(x[i]);
    }
    return result;
}