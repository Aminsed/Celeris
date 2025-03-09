#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <iostream>

namespace celeris {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64
};

class TensorImpl;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CUDA);
    Tensor(const void* data, const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CUDA);
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor();
    
    // Basic properties
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t size() const;
    DataType dtype() const;
    DeviceType device() const;
    
    // Data access
    void* data();
    const void* data() const;
    
    // Move to device
    Tensor to(DeviceType device) const;
    
    // Convert to different data type
    Tensor to(DataType dtype) const;
    
    // Gradient-related methods
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    Tensor& grad();
    const Tensor& grad() const;
    
    // Backward pass
    void backward();
    
    // Utility methods
    std::string to_string() const;
    
    // Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // In-place operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // Implementation details
    std::shared_ptr<TensorImpl> impl() const;
    
private:
    std::shared_ptr<TensorImpl> impl_;
};

// Stream operators
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// Create an aligned tensor with optimal memory access patterns
Tensor create_aligned_tensor(const std::vector<size_t>& shape, DataType dtype, DeviceType device);

} // namespace celeris 