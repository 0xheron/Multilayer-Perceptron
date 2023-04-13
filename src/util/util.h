#pragma once

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>

// A bunch of utitities for neural networks such as activation functions, loss functions, etc.
double relu(double x)
{
    return x > 0 ? x : 0;
}

Eigen::VectorX<double> relu(const Eigen::VectorX<double>& x)
{
    Eigen::VectorX<double> result(ct);
    for (int i = 0; i < result.size(); i++)
    {
        result[i] = relu(x[i]);
    }
    return result;
}