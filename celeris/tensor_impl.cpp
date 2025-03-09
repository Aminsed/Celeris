#include "celeris/tensor_impl.h"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>

namespace celeris {

// Constructors
TensorImpl::TensorImpl(const std::vector<size_t>& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), owns_data_(true), requires_grad_(false) {
    compute_strides();
    compute_num_elements();
    allocate();
}

TensorImpl::TensorImpl(const void* data, const std::vector<size_t>& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device), data_(nullptr), owns_data_(true), requires_grad_(false) {
    compute_strides();
    compute_num_elements();
    allocate();
    
    // Copy data to the tensor
    if (data) {
        copy_from(data, num_elements_ * element_size());
    }
}

// Copy constructor
TensorImpl::TensorImpl(const TensorImpl& other)
    : shape_(other.shape_), strides_(other.strides_), num_elements_(other.num_elements_),
      dtype_(other.dtype_), device_(other.device_), data_(nullptr),
      owns_data_(true), requires_grad_(other.requires_grad_) {
    allocate();
    
    // Copy data from the other tensor
    if (other.data_) {
        copy_from(other.data_, num_elements_ * element_size());
    }
    
    // Copy gradient if it exists
    if (other.grad_) {
        grad_ = std::make_shared<TensorImpl>(*other.grad_);
    }
}

// Move constructor
TensorImpl::TensorImpl(TensorImpl&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), num_elements_(other.num_elements_),
      dtype_(other.dtype_), device_(other.device_), data_(other.data_),
      owns_data_(other.owns_data_), requires_grad_(other.requires_grad_), grad_(std::move(other.grad_)) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

// Copy assignment operator
TensorImpl& TensorImpl::operator=(const TensorImpl& other) {
    if (this != &other) {
        // Clean up existing data
        deallocate();
        
        // Copy properties
        shape_ = other.shape_;
        strides_ = other.strides_;
        num_elements_ = other.num_elements_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        
        // Allocate and copy data
        allocate();
        if (other.data_) {
            copy_from(other.data_, num_elements_ * element_size());
        }
        
        // Copy gradient if it exists
        if (other.grad_) {
            grad_ = std::make_shared<TensorImpl>(*other.grad_);
        } else {
            grad_ = nullptr;
        }
    }
    return *this;
}

// Move assignment operator
TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept {
    if (this != &other) {
        // Clean up existing data
        deallocate();
        
        // Move properties
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        num_elements_ = other.num_elements_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        
        // Reset the other tensor
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

// Destructor
TensorImpl::~TensorImpl() {
    deallocate();
}

// Element size
size_t TensorImpl::element_size() const {
    switch (dtype_) {
        case DataType::FLOAT32:
            return sizeof(float);
        case DataType::FLOAT16:
            return sizeof(short); // half precision is 16 bits
        case DataType::INT32:
            return sizeof(int);
        case DataType::INT64:
            return sizeof(long long);
        default:
            return 0;
    }
}

// Memory management
void TensorImpl::allocate() {
    if (num_elements_ == 0) {
        return;
    }
    
    size_t size = num_elements_ * element_size();
    
    if (device_ == DeviceType::CUDA) {
        cudaError_t error = cudaMalloc(&data_, size);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory: " + std::string(cudaGetErrorString(error)));
        }
    } else {
        data_ = malloc(size);
        if (!data_) {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
    }
    
    owns_data_ = true;
}

void TensorImpl::deallocate() {
    if (data_ && owns_data_) {
        if (device_ == DeviceType::CUDA) {
            cudaFree(data_);
        } else {
            free(data_);
        }
        data_ = nullptr;
    }
}

void TensorImpl::copy_from(const void* src, size_t size) {
    if (!data_ || !src || size == 0) {
        return;
    }
    
    if (device_ == DeviceType::CUDA) {
        cudaError_t error = cudaMemcpy(data_, src, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to CUDA memory: " + std::string(cudaGetErrorString(error)));
        }
    } else {
        std::memcpy(data_, src, size);
    }
}

void TensorImpl::copy_to(void* dst, size_t size) const {
    if (!data_ || !dst || size == 0) {
        return;
    }
    
    if (device_ == DeviceType::CUDA) {
        cudaError_t error = cudaMemcpy(dst, data_, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from CUDA memory: " + std::string(cudaGetErrorString(error)));
        }
    } else {
        std::memcpy(dst, data_, size);
    }
}

// Gradient-related methods
void TensorImpl::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
    
    // Create a gradient tensor if needed
    if (requires_grad && !grad_) {
        grad_ = std::make_shared<TensorImpl>(shape_, dtype_, device_);
    }
}

// Utility methods
std::string TensorImpl::to_string() const {
    std::stringstream ss;
    
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        ss << shape_[i];
        if (i < shape_.size() - 1) {
            ss << ", ";
        }
    }
    ss << "], dtype=";
    
    switch (dtype_) {
        case DataType::FLOAT32:
            ss << "float32";
            break;
        case DataType::FLOAT16:
            ss << "float16";
            break;
        case DataType::INT32:
            ss << "int32";
            break;
        case DataType::INT64:
            ss << "int64";
            break;
    }
    
    ss << ", device=";
    ss << (device_ == DeviceType::CUDA ? "cuda" : "cpu");
    
    if (requires_grad_) {
        ss << ", requires_grad=True";
    }
    
    ss << ")";
    
    return ss.str();
}

// Helper methods
void TensorImpl::compute_strides() {
    strides_.resize(shape_.size());
    
    if (shape_.empty()) {
        return;
    }
    
    strides_[shape_.size() - 1] = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

void TensorImpl::compute_num_elements() {
    if (shape_.empty()) {
        num_elements_ = 0;
        return;
    }
    
    num_elements_ = 1;
    for (size_t dim : shape_) {
        num_elements_ *= dim;
    }
}

} // namespace celeris 