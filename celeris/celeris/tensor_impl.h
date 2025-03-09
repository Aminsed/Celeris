#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "celeris/tensor.h"

namespace celeris {

class TensorImpl {
public:
    // Constructors
    TensorImpl(const std::vector<size_t>& shape, DataType dtype, DeviceType device);
    TensorImpl(const void* data, const std::vector<size_t>& shape, DataType dtype, DeviceType device);
    
    // Copy and move constructors
    TensorImpl(const TensorImpl& other);
    TensorImpl(TensorImpl&& other) noexcept;
    
    // Assignment operators
    TensorImpl& operator=(const TensorImpl& other);
    TensorImpl& operator=(TensorImpl&& other) noexcept;
    
    // Destructor
    ~TensorImpl();
    
    // Basic properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return num_elements_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    size_t element_size() const;
    
    // Data access
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    // Memory management
    void allocate();
    void deallocate();
    void copy_from(const void* src, size_t size);
    void copy_to(void* dst, size_t size) const;
    
    // Gradient-related methods
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad);
    std::shared_ptr<TensorImpl> grad() const { return grad_; }
    void set_grad(std::shared_ptr<TensorImpl> grad) { grad_ = grad; }
    
    // Utility methods
    std::string to_string() const;
    
private:
    // Shape and size information
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t num_elements_;
    
    // Data type and device
    DataType dtype_;
    DeviceType device_;
    
    // Data storage
    void* data_;
    bool owns_data_;
    
    // Gradient information
    bool requires_grad_;
    std::shared_ptr<TensorImpl> grad_;
    
    // Helper methods
    void compute_strides();
    void compute_num_elements();
};

} // namespace celeris 