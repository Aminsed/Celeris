#include "celeris/cuda/jit.h"
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <cstring>
#include <cuda_runtime.h>  // For dim3
#include <nvrtc.h>         // For NVRTC
#include <cuda.h>          // For CUDA driver API

namespace celeris {
namespace cuda {

// Helper function to check NVRTC errors
#define CHECK_NVRTC(call) \
    do { \
        nvrtcResult result = call; \
        if (result != NVRTC_SUCCESS) { \
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("NVRTC error"); \
        } \
    } while(0)

// Helper function to check CUDA driver API errors
#define CHECK_CUDA(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* error_string; \
            cuGetErrorString(result, &error_string); \
            std::cerr << "CUDA error: " << error_string << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Kernel cache implementation
class KernelCache {
public:
    KernelCache() = default;
    ~KernelCache() = default;
    
    // Get a kernel from the cache or return nullptr if not found
    std::shared_ptr<CompiledKernel> get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    // Add a kernel to the cache
    void add(const std::string& key, std::shared_ptr<CompiledKernel> kernel) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[key] = kernel;
    }
    
private:
    std::unordered_map<std::string, std::shared_ptr<CompiledKernel>> cache_;
    std::mutex mutex_;
};

// CompiledKernel implementation
CompiledKernel::CompiledKernel(CUmodule module, CUfunction function)
    : module_(module), function_(function) {}

CompiledKernel::~CompiledKernel() {
    if (module_) {
        cuModuleUnload(module_);
    }
}

void CompiledKernel::launch(CUstream stream, void** args, dim3 grid, dim3 block) {
    if (!function_) {
        throw std::runtime_error("Cannot launch null kernel function");
    }
    
    // Calculate shared memory size (0 for now)
    unsigned int shared_mem_bytes = 0;
    
    // Launch the kernel
    CHECK_CUDA(cuLaunchKernel(
        function_,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        shared_mem_bytes,
        stream,
        args,
        nullptr
    ));
}

// JitCompiler implementation
JitCompiler::JitCompiler() : cache_(std::make_shared<KernelCache>()) {
    // Initialize CUDA driver API
    CHECK_CUDA(cuInit(0));
    
    // Get a CUDA device
    CHECK_CUDA(cuDeviceGet(&device_, 0));
    
    // Create a CUDA context
    CHECK_CUDA(cuCtxCreate(&context_, 0, device_));
}

JitCompiler::~JitCompiler() {
    if (context_) {
        cuCtxDestroy(context_);
    }
}

std::shared_ptr<CompiledKernel> JitCompiler::compile(const std::string& code, const JitCompileParams& params) {
    // Preprocess the code
    std::string processed_code = preprocess_code(code, params);
    
    // Create an NVRTC program
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(
        &prog,
        processed_code.c_str(),
        params.kernel_name.c_str(),
        0,
        nullptr,
        nullptr
    ));
    
    // Set compilation options
    std::vector<const char*> opts;
    
    // Add architecture-specific options
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Determine the compute capability
    int major = props.major;
    int minor = props.minor;
    
    // Set architecture-specific compilation flags
    std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    opts.push_back(arch.c_str());
    
    // Add optimization flags
    if (params.use_fast_math) {
        opts.push_back("--use_fast_math");
    }
    
    // Add more aggressive optimization flags from DeepEP
    opts.push_back("--std=c++17");  // Use newer C++ standard
    opts.push_back("-default-device");
    opts.push_back("-dopt=on");
    opts.push_back("-dlto=on");  // Link-time optimization
    opts.push_back("-dscalar-load-store=on");
    
    // Additional optimization flags for better performance
    opts.push_back("--restrict");  // Enable restrict keyword
    opts.push_back("-ftz=true");  // Flush denormals to zero
    opts.push_back("-prec-div=false");  // Fast division
    opts.push_back("-prec-sqrt=false");  // Fast square root
    
    // Architecture-specific optimizations
    if (major >= 8) {  // Ampere or newer
        opts.push_back("-dtensor-cores=on");
        // Enable FP8 support for Ampere and newer if available
        if (major == 9 || (major == 8 && minor >= 9)) {
            opts.push_back("-dfp8-support=on");
        }
    } else if (major == 7) {  // Volta/Turing
        opts.push_back("-dtensor-cores=on");
    }
    
    // Add user-specified options
    for (const auto& opt : params.compile_options) {
        opts.push_back(opt.c_str());
    }
    
    // Compile the program
    nvrtcResult compile_result = nvrtcCompileProgram(prog, opts.size(), opts.data());
    
    // Get compilation log
    size_t log_size;
    CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log(log_size, ' ');
    CHECK_NVRTC(nvrtcGetProgramLog(prog, &log[0]));
    
    if (compile_result != NVRTC_SUCCESS) {
        std::cerr << "NVRTC compilation failed: " << log << std::endl;
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("NVRTC compilation failed");
    }
    
    // Get PTX from the program
    size_t ptx_size;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, ' ');
    CHECK_NVRTC(nvrtcGetPTX(prog, &ptx[0]));
    
    // Clean up the NVRTC program
    CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    
    // Load the PTX as a CUDA module
    CUmodule module;
    
    // Set JIT options for the module
    CUjit_option jit_options[] = {
        CU_JIT_MAX_REGISTERS,
        CU_JIT_THREADS_PER_BLOCK,
        CU_JIT_OPTIMIZATION_LEVEL,
        CU_JIT_CACHE_MODE,
        CU_JIT_FTZ
    };
    
    // Set JIT option values
    void* jit_option_values[] = {
        (void*)(intptr_t)64,  // Max registers per thread
        (void*)(intptr_t)256, // Threads per block
        (void*)(intptr_t)4,   // Optimization level (max)
        (void*)(intptr_t)CU_JIT_CACHE_OPTION_CA, // Prefer L1 cache
        (void*)(intptr_t)1    // Flush denormals to zero
    };
    
    CHECK_CUDA(cuModuleLoadDataEx(&module, ptx.c_str(), 5, jit_options, jit_option_values));
    
    // Get the kernel function from the module
    CUfunction function;
    CHECK_CUDA(cuModuleGetFunction(&function, module, params.kernel_name.c_str()));
    
    // Create and return the compiled kernel
    return std::make_shared<CompiledKernel>(module, function);
}

std::shared_ptr<CompiledKernel> JitCompiler::get_or_compile(const std::string& code, const JitCompileParams& params) {
    // Get the kernel hash
    std::string key = get_kernel_hash(code, params);
    
    // Try to get the kernel from the cache
    auto kernel = cache_->get(key);
    if (kernel) {
        return kernel;
    }
    
    // Compile the kernel
    kernel = compile(code, params);
    
    // Add the kernel to the cache
    cache_->add(key, kernel);
    
    return kernel;
}

std::string JitCompiler::preprocess_code(const std::string& code, const JitCompileParams& params) {
    // Add any necessary includes or defines
    std::stringstream ss;
    
    // Add includes
    ss << "#include <cuda_runtime.h>\n";
    ss << "#include <cuda_fp16.h>\n";
    
    // Add tensor core includes if needed
    if (params.use_tensor_cores) {
        ss << "#include <mma.h>\n";
        ss << "using namespace nvcuda;\n";
    }
    
    // Add common defines
    ss << "#define BLOCK_SIZE_M " << params.block_size_m << "\n";
    ss << "#define BLOCK_SIZE_N " << params.block_size_n << "\n";
    ss << "#define BLOCK_SIZE_K " << params.block_size_k << "\n";
    
    // Add unaligned blocks define if needed
    if (params.use_unaligned_blocks) {
        ss << "#define USE_UNALIGNED_BLOCKS 1\n";
    } else {
        ss << "#define USE_UNALIGNED_BLOCKS 0\n";
    }
    
    // Add any custom defines from params
    for (const auto& define : params.defines) {
        ss << "#define " << define.first << " " << define.second << "\n";
    }
    
    // Add the original code
    ss << code;
    
    return ss.str();
}

std::string JitCompiler::get_kernel_hash(const std::string& code, const JitCompileParams& params) {
    // Create a hash based on the code and parameters
    std::stringstream ss;
    ss << params.kernel_name;
    
    // Add key parameters to the hash
    ss << "_m" << params.block_size_m;
    ss << "_n" << params.block_size_n;
    ss << "_k" << params.block_size_k;
    ss << "_ua" << (params.use_unaligned_blocks ? "1" : "0");
    ss << "_tc" << (params.use_tensor_cores ? "1" : "0");
    ss << "_fm" << (params.use_fast_math ? "1" : "0");
    
    // Add defines to the hash
    for (const auto& define : params.defines) {
        ss << "_" << define.first << "_" << define.second;
    }
    
    // Add a hash of the code
    std::hash<std::string> hasher;
    ss << "_" << hasher(code);
    
    return ss.str();
}

// Singleton JIT compiler
JitCompiler& get_jit_compiler() {
    static JitCompiler compiler;
    return compiler;
}

} // namespace cuda
} // namespace celeris 