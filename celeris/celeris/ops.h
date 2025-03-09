#pragma once

#include "celeris/tensor.h"

namespace celeris {

// Basic operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Scalar operations
Tensor add(const Tensor& a, float scalar);
Tensor sub(const Tensor& a, float scalar);
Tensor mul(const Tensor& a, float scalar);
Tensor div(const Tensor& a, float scalar);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a, const std::vector<size_t>& dims = {});

// Reduction operations
Tensor mean(const Tensor& x, int dim = -1);

// Activation functions
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor softmax(const Tensor& x, int dim = -1);

// Loss functions
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target);

// Neural network layers
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor());
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor(), 
              const std::vector<size_t>& stride = {1, 1}, 
              const std::vector<size_t>& padding = {0, 0}, 
              const std::vector<size_t>& dilation = {1, 1});
Tensor max_pool2d(const Tensor& input, 
                 const std::vector<size_t>& kernel_size, 
                 const std::vector<size_t>& stride = {}, 
                 const std::vector<size_t>& padding = {0, 0});
Tensor batch_norm2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const Tensor& running_mean, const Tensor& running_var,
                   bool training = true, double momentum = 0.1, double eps = 1e-5);

// Utility functions
Tensor zeros(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CUDA);
Tensor ones(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CUDA);
Tensor randn(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CUDA);
Tensor from_numpy(const void* data, const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32);

} // namespace celeris 