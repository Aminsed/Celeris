#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "celeris/tensor.h"

namespace celeris {
namespace cuda {

// GEMM (General Matrix Multiplication) kernel with DeepGEMM optimizations
// C = alpha * A * B + beta * C
void gemm(const void* A, const void* B, void* C,
          int M, int N, int K,
          DataType dtype,
          float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = nullptr);

// Element-wise operations
void add(const void* a, const void* b, void* c,
         size_t size, DataType dtype,
         cudaStream_t stream = nullptr);

void sub(const void* a, const void* b, void* c,
         size_t size, DataType dtype,
         cudaStream_t stream = nullptr);

void mul(const void* a, const void* b, void* c,
         size_t size, DataType dtype,
         cudaStream_t stream = nullptr);

void div(const void* a, const void* b, void* c,
         size_t size, DataType dtype,
         cudaStream_t stream = nullptr);

// Activation functions
void relu(const void* x, void* y,
          size_t size, DataType dtype,
          cudaStream_t stream = nullptr);

void relu_backward(const void* dy, const void* y, void* dx,
                  size_t size, DataType dtype,
                  cudaStream_t stream = nullptr);

void sigmoid(const void* x, void* y,
             size_t size, DataType dtype,
             cudaStream_t stream = nullptr);

void sigmoid_backward(const void* dy, const void* y, void* dx,
                     size_t size, DataType dtype,
                     cudaStream_t stream = nullptr);

void tanh(const void* x, void* y,
          size_t size, DataType dtype,
          cudaStream_t stream = nullptr);

void tanh_backward(const void* dy, const void* y, void* dx,
                  size_t size, DataType dtype,
                  cudaStream_t stream = nullptr);

void softmax(const void* x, void* y,
             const std::vector<size_t>& shape, int dim,
             DataType dtype,
             cudaStream_t stream = nullptr);

void softmax_backward(const void* dy, const void* y, void* dx,
                     const std::vector<size_t>& shape, int dim,
                     DataType dtype,
                     cudaStream_t stream = nullptr);

// Loss functions
void mse_loss(const void* pred, const void* target, void* loss,
              size_t size, DataType dtype,
              cudaStream_t stream = nullptr);

void mse_loss_backward(const void* pred, const void* target, void* grad,
                      size_t size, DataType dtype,
                      cudaStream_t stream = nullptr);

void cross_entropy_loss(const void* pred, const void* target, void* loss,
                       const std::vector<size_t>& shape, int dim,
                       DataType dtype,
                       cudaStream_t stream = nullptr);

void cross_entropy_loss_backward(const void* pred, const void* target, void* grad,
                               const std::vector<size_t>& shape, int dim,
                               DataType dtype,
                               cudaStream_t stream = nullptr);

// Utility functions
void fill(void* data, float value, size_t size, DataType dtype,
          cudaStream_t stream = nullptr);

void randn(void* data, size_t size, DataType dtype,
           unsigned long long seed,
           cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace celeris 