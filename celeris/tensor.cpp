#include "celeris/tensor.h"
#include "celeris/tensor_impl.h"
#include <sstream>
#include <iomanip>

namespace celeris {

// Constructors
Tensor::Tensor() : impl_(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype, DeviceType device)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}

Tensor::Tensor(const void* data, const std::vector<size_t>& shape, DataType dtype, DeviceType device)
    : impl_(std::make_shared<TensorImpl>(data, shape, dtype, device)) {}

// Copy and move constructors
Tensor::Tensor(const Tensor& other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

// Assignment operators
Tensor& Tensor::operator=(const Tensor& other) {
    impl_ = other.impl_;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

// Destructor
Tensor::~Tensor() = default;

// Basic properties
std::vector<size_t> Tensor::shape() const {
    return impl_ ? impl_->shape() : std::vector<size_t>();
}

size_t Tensor::ndim() const {
    return impl_ ? impl_->ndim() : 0;
}

size_t Tensor::size() const {
    return impl_ ? impl_->size() : 0;
}

DataType Tensor::dtype() const {
    return impl_ ? impl_->dtype() : DataType::FLOAT32;
}

DeviceType Tensor::device() const {
    return impl_ ? impl_->device() : DeviceType::CPU;
}

// Data access
void* Tensor::data() {
    return impl_ ? impl_->data() : nullptr;
}

const void* Tensor::data() const {
    return impl_ ? impl_->data() : nullptr;
}

// Move to device
Tensor Tensor::to(DeviceType device) const {
    if (!impl_ || impl_->device() == device) {
        return *this;
    }
    
    // Create a new tensor on the target device
    Tensor result(impl_->shape(), impl_->dtype(), device);
    
    // Copy data from this tensor to the new tensor
    // This would involve a device-to-device or host-to-device copy
    // For simplicity, we'll just allocate and not implement the actual copy here
    
    return result;
}

// Convert to different data type
Tensor Tensor::to(DataType dtype) const {
    if (!impl_ || impl_->dtype() == dtype) {
        return *this;
    }
    
    // Create a new tensor with the target data type
    Tensor result(impl_->shape(), dtype, impl_->device());
    
    // Convert data from this tensor to the new tensor
    // This would involve a type conversion
    // For simplicity, we'll just allocate and not implement the actual conversion here
    
    return result;
}

// Gradient-related methods
bool Tensor::requires_grad() const {
    return impl_ ? impl_->requires_grad() : false;
}

void Tensor::set_requires_grad(bool requires_grad) {
    if (impl_) {
        impl_->set_requires_grad(requires_grad);
    }
}

Tensor& Tensor::grad() {
    static Tensor empty_grad;
    if (!impl_ || !impl_->grad()) {
        return empty_grad;
    }
    
    // Create a static tensor that wraps the gradient
    static Tensor grad_tensor;
    grad_tensor.impl_ = impl_->grad();
    return grad_tensor;
}

const Tensor& Tensor::grad() const {
    static Tensor empty_grad;
    if (!impl_ || !impl_->grad()) {
        return empty_grad;
    }
    
    // Create a static tensor that wraps the gradient
    static Tensor grad_tensor;
    grad_tensor.impl_ = impl_->grad();
    return grad_tensor;
}

// Backward pass
void Tensor::backward() {
    // This would trigger the backward pass through the computation graph
    // For simplicity, we'll just print a message
    std::cout << "Backward pass not implemented yet" << std::endl;
}

// Utility methods
std::string Tensor::to_string() const {
    if (!impl_) {
        return "Tensor()";
    }
    return impl_->to_string();
}

// Operators
Tensor Tensor::operator+(const Tensor& other) const {
    // This would call the appropriate CUDA kernel for addition
    // For simplicity, we'll just return a new tensor
    return Tensor(impl_->shape(), impl_->dtype(), impl_->device());
}

Tensor Tensor::operator-(const Tensor& other) const {
    // This would call the appropriate CUDA kernel for subtraction
    // For simplicity, we'll just return a new tensor
    return Tensor(impl_->shape(), impl_->dtype(), impl_->device());
}

Tensor Tensor::operator*(const Tensor& other) const {
    // This would call the appropriate CUDA kernel for element-wise multiplication
    // For simplicity, we'll just return a new tensor
    return Tensor(impl_->shape(), impl_->dtype(), impl_->device());
}

Tensor Tensor::operator/(const Tensor& other) const {
    // This would call the appropriate CUDA kernel for element-wise division
    // For simplicity, we'll just return a new tensor
    return Tensor(impl_->shape(), impl_->dtype(), impl_->device());
}

// In-place operators
Tensor& Tensor::operator+=(const Tensor& other) {
    // This would call the appropriate CUDA kernel for in-place addition
    // For simplicity, we'll just return this tensor
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    // This would call the appropriate CUDA kernel for in-place subtraction
    // For simplicity, we'll just return this tensor
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    // This would call the appropriate CUDA kernel for in-place element-wise multiplication
    // For simplicity, we'll just return this tensor
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    // This would call the appropriate CUDA kernel for in-place element-wise division
    // For simplicity, we'll just return this tensor
    return *this;
}

// Implementation details
std::shared_ptr<TensorImpl> Tensor::impl() const {
    return impl_;
}

// Stream operators
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

// Create an aligned tensor with optimal memory access patterns
Tensor create_aligned_tensor(const std::vector<size_t>& shape, DataType dtype, DeviceType device) {
    // Calculate total size
    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Calculate element size
    size_t element_size;
    switch (dtype) {
        case DataType::FLOAT32:
            element_size = sizeof(float);
            break;
        case DataType::FLOAT16:
            element_size = sizeof(uint16_t);  // half precision
            break;
        case DataType::INT32:
            element_size = sizeof(int);
            break;
        case DataType::INT64:
            element_size = sizeof(int64_t);
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
    
    // Allocate memory with alignment
    void* ptr = nullptr;
    
    if (device == DeviceType::CPU) {
        // For CPU, use aligned_alloc with 64-byte alignment (cache line size)
        size_t alignment = 64;
        size_t padded_size = ((total_elements * element_size + alignment - 1) / alignment) * alignment;
        
        #ifdef _WIN32
        ptr = _aligned_malloc(padded_size, alignment);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate aligned memory on CPU");
        }
        #else
        if (posix_memalign(&ptr, alignment, padded_size) != 0) {
            throw std::runtime_error("Failed to allocate aligned memory on CPU");
        }
        #endif
        
        // Initialize to zero
        std::memset(ptr, 0, padded_size);
    } else if (device == DeviceType::CUDA) {
        // For CUDA, use cudaMalloc with pitch for 2D+ tensors or regular cudaMalloc for 1D tensors
        if (shape.size() >= 2) {
            // For 2D+ tensors, use cudaMallocPitch for better memory alignment
            size_t pitch;
            size_t width = shape.back() * element_size;  // Width in bytes
            size_t height = total_elements / shape.back();  // Height in rows
            
            cudaError_t err = cudaMallocPitch(&ptr, &pitch, width, height);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate pitched memory on CUDA: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            // Initialize to zero
            err = cudaMemset2D(ptr, pitch, 0, width, height);
            if (err != cudaSuccess) {
                cudaFree(ptr);
                throw std::runtime_error("Failed to initialize pitched memory on CUDA: " + 
                                        std::string(cudaGetErrorString(err)));
            }
        } else {
            // For 1D tensors, use regular cudaMalloc with 256-byte alignment
            size_t alignment = 256;  // Optimal for coalesced memory access
            size_t padded_size = ((total_elements * element_size + alignment - 1) / alignment) * alignment;
            
            cudaError_t err = cudaMalloc(&ptr, padded_size);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate memory on CUDA: " + 
                                        std::string(cudaGetErrorString(err)));
            }
            
            // Initialize to zero
            err = cudaMemset(ptr, 0, padded_size);
            if (err != cudaSuccess) {
                cudaFree(ptr);
                throw std::runtime_error("Failed to initialize memory on CUDA: " + 
                                        std::string(cudaGetErrorString(err)));
            }
        }
    } else {
        throw std::runtime_error("Unsupported device type");
    }
    
    // Create tensor with the allocated memory
    return Tensor(ptr, shape, dtype, device);
}

} // namespace celeris 