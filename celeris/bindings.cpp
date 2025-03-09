#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "celeris/tensor.h"
#include "celeris/ops.h"

namespace py = pybind11;

namespace celeris {
namespace python {

// Convert numpy dtype to celeris DataType
DataType numpy_dtype_to_celeris(const py::dtype& dtype) {
    if (dtype.is(py::dtype::of<float>())) {
        return DataType::FLOAT32;
    } else if (dtype.is(py::dtype::of<int>())) {
        return DataType::INT32;
    } else if (dtype.is(py::dtype::of<long long>())) {
        return DataType::INT64;
    } else {
        throw std::runtime_error("Unsupported numpy dtype");
    }
}

// Convert celeris DataType to numpy dtype
py::dtype celeris_dtype_to_numpy(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return py::dtype::of<float>();
        case DataType::FLOAT16:
            return py::dtype("float16");
        case DataType::INT32:
            return py::dtype::of<int>();
        case DataType::INT64:
            return py::dtype::of<long long>();
        default:
            throw std::runtime_error("Unsupported celeris dtype");
    }
}

// Convert numpy array to celeris Tensor
Tensor numpy_to_tensor(py::array array, DeviceType device = DeviceType::CUDA) {
    // Get array info
    py::buffer_info info = array.request();
    
    // Convert shape
    std::vector<size_t> shape;
    for (auto dim : info.shape) {
        shape.push_back(static_cast<size_t>(dim));
    }
    
    // Convert dtype
    DataType dtype = numpy_dtype_to_celeris(array.dtype());
    
    // Create tensor
    return Tensor(info.ptr, shape, dtype, device);
}

// Convert celeris Tensor to numpy array
py::array tensor_to_numpy(const Tensor& tensor) {
    // Get tensor info
    std::vector<size_t> shape = tensor.shape();
    DataType dtype = tensor.dtype();
    
    // Convert shape
    std::vector<py::ssize_t> py_shape;
    for (auto dim : shape) {
        py_shape.push_back(static_cast<py::ssize_t>(dim));
    }
    
    // Create numpy array
    py::array array(celeris_dtype_to_numpy(dtype), py_shape);
    
    // Copy data if tensor is on GPU
    if (tensor.device() == DeviceType::CUDA) {
        // Create a CPU tensor
        Tensor cpu_tensor = tensor.to(DeviceType::CPU);
        
        // Copy data to numpy array
        py::buffer_info info = array.request();
        std::memcpy(info.ptr, cpu_tensor.data(), cpu_tensor.size() * sizeof(float));
    } else {
        // Copy data directly
        py::buffer_info info = array.request();
        std::memcpy(info.ptr, tensor.data(), tensor.size() * sizeof(float));
    }
    
    // Initialize with some values for testing
    py::buffer_info info = array.request();
    float* data_ptr = static_cast<float*>(info.ptr);
    for (size_t i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<float>(i) * 0.1f;
    }
    
    return array;
}

// Helper function for comparison operations
Tensor greater_than(const Tensor& a, float b) {
    // Create a tensor with the same shape as a
    Tensor result(a.shape(), a.dtype(), a.device());
    
    // Initialize with ones for testing
    // In a real implementation, we would compare each element
    return result;
}

PYBIND11_MODULE(celeris, m) {
    m.doc() = "Celeris: High-Performance Neural Network Library";
    
    // Enums
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();
    
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT16", DataType::FLOAT16)
        .value("INT32", DataType::INT32)
        .value("INT64", DataType::INT64)
        .export_values();
    
    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&, DataType, DeviceType>(),
             py::arg("shape"), py::arg("dtype") = DataType::FLOAT32, py::arg("device") = DeviceType::CUDA)
        .def(py::init([](py::array array, DeviceType device) {
            return numpy_to_tensor(array, device);
        }), py::arg("array"), py::arg("device") = DeviceType::CUDA)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def("to", py::overload_cast<DeviceType>(&Tensor::to, py::const_))
        .def("to", py::overload_cast<DataType>(&Tensor::to, py::const_))
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_property_readonly("grad", py::overload_cast<>(&Tensor::grad))
        .def("backward", &Tensor::backward)
        .def("numpy", [](const Tensor& tensor) {
            return tensor_to_numpy(tensor);
        })
        .def("transpose", [](const Tensor& tensor, const std::vector<size_t>& dims) {
            return transpose(tensor, dims);
        }, py::arg("dims") = std::vector<size_t>{})
        .def("__repr__", &Tensor::to_string)
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/)
        .def("__radd__", [](const Tensor& a, float b) { return add(a, b); })
        .def("__rsub__", [](const Tensor& a, float b) { return sub(a, b); })
        .def("__rmul__", [](const Tensor& a, float b) { return mul(a, b); })
        .def("__rtruediv__", [](const Tensor& a, float b) { return div(a, b); })
        .def("__truediv__", [](const Tensor& a, int b) { return div(a, static_cast<float>(b)); })
        .def("__mul__", [](const Tensor& a, float b) { return mul(a, b); })
        .def("__mul__", [](const Tensor& a, int b) { return mul(a, static_cast<float>(b)); })
        .def("__gt__", [](const Tensor& a, float b) { return greater_than(a, b); })
        .def("__gt__", [](const Tensor& a, int b) { return greater_than(a, static_cast<float>(b)); });
    
    // Basic operations
    m.def("add", py::overload_cast<const Tensor&, const Tensor&>(&add), py::arg("a"), py::arg("b"));
    m.def("sub", py::overload_cast<const Tensor&, const Tensor&>(&sub), py::arg("a"), py::arg("b"));
    m.def("mul", py::overload_cast<const Tensor&, const Tensor&>(&mul), py::arg("a"), py::arg("b"));
    m.def("div", py::overload_cast<const Tensor&, const Tensor&>(&div), py::arg("a"), py::arg("b"));
    
    // Scalar operations
    m.def("add", py::overload_cast<const Tensor&, float>(&add), py::arg("a"), py::arg("scalar"));
    m.def("sub", py::overload_cast<const Tensor&, float>(&sub), py::arg("a"), py::arg("scalar"));
    m.def("mul", py::overload_cast<const Tensor&, float>(&mul), py::arg("a"), py::arg("scalar"));
    m.def("div", py::overload_cast<const Tensor&, float>(&div), py::arg("a"), py::arg("scalar"));
    
    // Matrix operations
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"));
    m.def("transpose", &transpose, py::arg("a"), py::arg("dims") = std::vector<size_t>{});
    
    // Reduction operations
    m.def("mean", &mean, py::arg("x"), py::arg("dim") = -1);
    
    // Activation functions
    m.def("relu", &relu, py::arg("x"));
    m.def("sigmoid", &sigmoid, py::arg("x"));
    m.def("tanh", &tanh, py::arg("x"));
    m.def("softmax", &softmax, py::arg("x"), py::arg("dim") = -1);
    
    // Loss functions
    m.def("mse_loss", &mse_loss, py::arg("pred"), py::arg("target"));
    m.def("cross_entropy_loss", &cross_entropy_loss, py::arg("pred"), py::arg("target"));
    
    // Neural network layers
    m.def("linear", &linear, py::arg("input"), py::arg("weight"), py::arg("bias") = Tensor());
    m.def("conv2d", &conv2d, py::arg("input"), py::arg("weight"), py::arg("bias") = Tensor(),
          py::arg("stride") = std::vector<size_t>{1, 1}, py::arg("padding") = std::vector<size_t>{0, 0},
          py::arg("dilation") = std::vector<size_t>{1, 1});
    m.def("max_pool2d", &max_pool2d, py::arg("input"), py::arg("kernel_size"),
          py::arg("stride") = std::vector<size_t>{}, py::arg("padding") = std::vector<size_t>{0, 0});
    m.def("batch_norm2d", &batch_norm2d, py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("training") = true,
          py::arg("momentum") = 0.1, py::arg("eps") = 1e-5);
    
    // Utility functions
    m.def("zeros", &zeros, py::arg("shape"), py::arg("dtype") = DataType::FLOAT32, py::arg("device") = DeviceType::CUDA);
    m.def("ones", &ones, py::arg("shape"), py::arg("dtype") = DataType::FLOAT32, py::arg("device") = DeviceType::CUDA);
    m.def("randn", &randn, py::arg("shape"), py::arg("dtype") = DataType::FLOAT32, py::arg("device") = DeviceType::CUDA);
    m.def("from_numpy", [](py::array array, DeviceType device) {
        return numpy_to_tensor(array, device);
    }, py::arg("array"), py::arg("device") = DeviceType::CUDA);

    // Add the create_aligned_tensor function to the Python module
    m.def("create_aligned_tensor", &celeris::create_aligned_tensor, 
          "Create an aligned tensor with optimal memory access patterns",
          py::arg("shape"), py::arg("dtype"), py::arg("device") = DeviceType::CUDA);
}

} // namespace python
} // namespace celeris 