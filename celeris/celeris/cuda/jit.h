#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>  // For dim3

namespace celeris {
namespace cuda {

// Forward declaration
class KernelCache;

// JIT compilation parameters
struct JitCompileParams {
    std::vector<std::string> compile_options;
    std::string kernel_name;
    bool use_fast_math = true;
    bool use_tensor_cores = true;
    int block_size_m = 128;
    int block_size_n = 128;
    int block_size_k = 32;
    bool use_unaligned_blocks = false;
    
    // Key-value pairs for #define directives
    std::unordered_map<std::string, std::string> defines;
};

// Compiled kernel
class CompiledKernel {
public:
    CompiledKernel(CUmodule module, CUfunction function);
    ~CompiledKernel();
    
    // Launch the kernel
    void launch(CUstream stream, void** args, dim3 grid, dim3 block);
    
    // Get the function
    CUfunction function() const { return function_; }
    
private:
    CUmodule module_;
    CUfunction function_;
};

// JIT compiler
class JitCompiler {
public:
    JitCompiler();
    ~JitCompiler();
    
    // Compile a kernel
    std::shared_ptr<CompiledKernel> compile(const std::string& code, const JitCompileParams& params);
    
    // Get a kernel from the cache or compile it
    std::shared_ptr<CompiledKernel> get_or_compile(const std::string& code, const JitCompileParams& params);
    
private:
    // Kernel cache
    std::shared_ptr<KernelCache> cache_;
    
    // CUDA device
    CUdevice device_;
    
    // CUDA context
    CUcontext context_;
    
    // Helper methods
    std::string preprocess_code(const std::string& code, const JitCompileParams& params);
    std::string get_kernel_hash(const std::string& code, const JitCompileParams& params);
};

// Singleton JIT compiler
JitCompiler& get_jit_compiler();

} // namespace cuda
} // namespace celeris 