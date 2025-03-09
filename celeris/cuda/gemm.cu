#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <sstream>
#include <regex>
#include "celeris/cuda/kernels.h"
#include "celeris/cuda/jit.h"

namespace celeris {
namespace cuda {

// Template for the GEMM kernel with persistent warp specialization
const char* GEMM_KERNEL_TEMPLATE = R"(
// GEMM kernel with DeepGEMM optimizations
// C = alpha * A * B + beta * C
// Control register usage with maxrregcount pragma
#pragma nv_diag_suppress 177  // Suppress unused variable warnings
extern "C" __global__ void __launch_bounds__(${block_threads}, ${min_blocks_per_sm}) gemm_kernel(
    const ${data_type}* __restrict__ A,
    const ${data_type}* __restrict__ B,
    ${data_type}* __restrict__ C,
    int M, int N, int K,
    ${data_type} alpha, ${data_type} beta) {
    
    // Grid-stride loop for efficient block scheduling
    // Each thread block processes multiple tiles in a grid-stride fashion
    for (int block_row = blockIdx.y; block_row < (M + ${block_size_m} - 1) / ${block_size_m}; block_row += gridDim.y) {
        for (int block_col = blockIdx.x; block_col < (N + ${block_size_n} - 1) / ${block_size_n}; block_col += gridDim.x) {
            // Thread indices
            int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id = thread_id / WARP_SIZE;
            int lane_id = thread_id % WARP_SIZE;
            
            // Warp row and column
            int warp_row = warp_id / ${warps_per_block_n};
            int warp_col = warp_id % ${warps_per_block_n};
            
            // Shared memory for tiles
            __shared__ ${data_type} tileA[2][${block_size_m}][${block_size_k}];
            __shared__ ${data_type} tileB[2][${block_size_k}][${block_size_n}];
            
            // Registers for accumulation - explicitly marked as register variables
            register ${data_type} accum[${warp_size_m}][${warp_size_n}];
            
            // Initialize accumulation registers to zero
            #pragma unroll
            for (int i = 0; i < ${warp_size_m}; ++i) {
                #pragma unroll
                for (int j = 0; j < ${warp_size_n}; ++j) {
                    accum[i][j] = 0;
                }
            }
            
            // Starting positions
            register int a_row_start = block_row * ${block_size_m};
            register int b_col_start = block_col * ${block_size_n};
            
            // Double buffering index
            register int buffer_idx = 0;
            
            // Prefetch first tiles - use vector loads where possible for better memory throughput
            #pragma unroll 4
            for (int i = thread_id; i < ${block_size_m} * ${block_size_k}; i += blockDim.x * blockDim.y) {
                register int row = i / ${block_size_k};
                register int col = i % ${block_size_k};
                register int a_row = a_row_start + row;
                if (a_row < M && col < K) {
                    tileA[buffer_idx][row][col] = A[a_row * K + col];
                } else {
                    tileA[buffer_idx][row][col] = 0;
                }
            }
            
            #pragma unroll 4
            for (int i = thread_id; i < ${block_size_k} * ${block_size_n}; i += blockDim.x * blockDim.y) {
                register int row = i / ${block_size_n};
                register int col = i % ${block_size_n};
                register int b_col = b_col_start + col;
                if (row < K && b_col < N) {
                    tileB[buffer_idx][row][col] = B[row * N + b_col];
                } else {
                    tileB[buffer_idx][row][col] = 0;
                }
            }
            
            __syncthreads();
            
            // Main loop over K dimension
            for (int k_tile = 0; k_tile < (K + ${block_size_k} - 1) / ${block_size_k}; ++k_tile) {
                // Next buffer index for double buffering
                register int next_buffer_idx = 1 - buffer_idx;
                
                // Prefetch next tiles if not the last iteration
                if (k_tile < (K + ${block_size_k} - 1) / ${block_size_k} - 1) {
                    register int next_k_offset = (k_tile + 1) * ${block_size_k};
                    
                    #pragma unroll 4
                    for (int i = thread_id; i < ${block_size_m} * ${block_size_k}; i += blockDim.x * blockDim.y) {
                        register int row = i / ${block_size_k};
                        register int col = i % ${block_size_k};
                        register int a_row = a_row_start + row;
                        register int a_col = next_k_offset + col;
                        if (a_row < M && a_col < K) {
                            tileA[next_buffer_idx][row][col] = A[a_row * K + a_col];
                        } else {
                            tileA[next_buffer_idx][row][col] = 0;
                        }
                    }
                    
                    #pragma unroll 4
                    for (int i = thread_id; i < ${block_size_k} * ${block_size_n}; i += blockDim.x * blockDim.y) {
                        register int row = i / ${block_size_n};
                        register int col = i % ${block_size_n};
                        register int b_row = next_k_offset + row;
                        register int b_col = b_col_start + col;
                        if (b_row < K && b_col < N) {
                            tileB[next_buffer_idx][row][col] = B[b_row * N + b_col];
                        } else {
                            tileB[next_buffer_idx][row][col] = 0;
                        }
                    }
                }
                
                // Compute matrix multiplication for this tile with enhanced ILP
                // Process multiple k elements at once when possible
                #pragma unroll 8  // More aggressive unrolling for better ILP
                for (int k = 0; k < ${block_size_k}; k += 4) {
                    // Check if we can process 4 elements at once
                    if (k + 3 < ${block_size_k}) {
                        // Load values from shared memory to registers using vector loads when possible
                        register ${data_type} a_reg[${warp_size_m}][4];  // Store 4 consecutive elements
                        register ${data_type} b_reg[${warp_size_n}][4];  // Store 4 consecutive elements
                        
                        // Load A values with interleaved pattern to reduce bank conflicts
                        #pragma unroll
                        for (int i = 0; i < ${warp_size_m}; ++i) {
                            register int a_row = warp_row * ${warp_size_m} + i;
                            a_reg[i][0] = tileA[buffer_idx][a_row][k];
                            a_reg[i][1] = tileA[buffer_idx][a_row][k+1];
                            a_reg[i][2] = tileA[buffer_idx][a_row][k+2];
                            a_reg[i][3] = tileA[buffer_idx][a_row][k+3];
                        }
                        
                        // Load B values with interleaved pattern to reduce bank conflicts
                        #pragma unroll
                        for (int i = 0; i < ${warp_size_n}; ++i) {
                            register int b_col = warp_col * ${warp_size_n} + i;
                            b_reg[i][0] = tileB[buffer_idx][k][b_col];
                            b_reg[i][1] = tileB[buffer_idx][k+1][b_col];
                            b_reg[i][2] = tileB[buffer_idx][k+2][b_col];
                            b_reg[i][3] = tileB[buffer_idx][k+3][b_col];
                        }
                        
                        // Perform matrix multiplication with interleaved operations
                        // This reduces dependency chains and improves instruction scheduling
                        #pragma unroll
                        for (int i = 0; i < ${warp_size_m}; ++i) {
                            #pragma unroll
                            for (int j = 0; j < ${warp_size_n}; ++j) {
                                // Interleaved operations for better instruction-level parallelism
                                accum[i][j] += a_reg[i][0] * b_reg[j][0];
                                accum[i][j] += a_reg[i][1] * b_reg[j][1];
                                accum[i][j] += a_reg[i][2] * b_reg[j][2];
                                accum[i][j] += a_reg[i][3] * b_reg[j][3];
                            }
                        }
                    } else {
                        // Handle remaining k elements (less than 4)
                        for (int k_offset = 0; k_offset < 4 && k + k_offset < ${block_size_k}; ++k_offset) {
                            // Load values from shared memory to registers
                            register ${data_type} a_reg[${warp_size_m}];
                            register ${data_type} b_reg[${warp_size_n}];
                            
                            #pragma unroll
                            for (int i = 0; i < ${warp_size_m}; ++i) {
                                register int a_row = warp_row * ${warp_size_m} + i;
                                a_reg[i] = tileA[buffer_idx][a_row][k + k_offset];
                            }
                            
                            #pragma unroll
                            for (int i = 0; i < ${warp_size_n}; ++i) {
                                register int b_col = warp_col * ${warp_size_n} + i;
                                b_reg[i] = tileB[buffer_idx][k + k_offset][b_col];
                            }
                            
                            // Perform matrix multiplication
                            #pragma unroll
                            for (int i = 0; i < ${warp_size_m}; ++i) {
                                #pragma unroll
                                for (int j = 0; j < ${warp_size_n}; ++j) {
                                    accum[i][j] += a_reg[i] * b_reg[j];
                                }
                            }
                        }
                    }
                }
                
                // Switch buffers
                buffer_idx = next_buffer_idx;
                
                // Synchronize before loading next tiles
                __syncthreads();
            }
            
            // Write results to global memory
            #pragma unroll
            for (int i = 0; i < ${warp_size_m}; ++i) {
                register int c_row = a_row_start + warp_row * ${warp_size_m} + i;
                if (c_row < M) {
                    #pragma unroll
                    for (int j = 0; j < ${warp_size_n}; ++j) {
                        register int c_col = b_col_start + warp_col * ${warp_size_n} + j;
                        if (c_col < N) {
                            register int c_idx = c_row * N + c_col;
                            if (beta == 0) {
                                C[c_idx] = alpha * accum[i][j];
                            } else {
                                C[c_idx] = alpha * accum[i][j] + beta * C[c_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}
)";

// Template for the GEMM kernel with tensor cores (for Ampere architecture and above)
const char* GEMM_TENSOR_CORE_TEMPLATE = R"(
// GEMM kernel with DeepGEMM optimizations and tensor cores
// C = alpha * A * B + beta * C
// Control register usage with maxrregcount pragma
#pragma nv_diag_suppress 177  // Suppress unused variable warnings
extern "C" __global__ void __launch_bounds__(${block_threads}, ${min_blocks_per_sm}) gemm_tensor_core_kernel(
    const ${data_type}* __restrict__ A,
    const ${data_type}* __restrict__ B,
    ${data_type}* __restrict__ C,
    int M, int N, int K,
    ${data_type} alpha, ${data_type} beta) {
    
    // Grid-stride loop for efficient block scheduling
    // Each thread block processes multiple tiles in a grid-stride fashion
    for (int block_row = blockIdx.y; block_row < (M + ${block_size_m} - 1) / ${block_size_m}; block_row += gridDim.y) {
        for (int block_col = blockIdx.x; block_col < (N + ${block_size_n} - 1) / ${block_size_n}; block_col += gridDim.x) {
            // Thread indices
            int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id = thread_id / WARP_SIZE;
            int lane_id = thread_id % WARP_SIZE;
            
            // Warp row and column
            int warp_row = warp_id / ${warps_per_block_n};
            int warp_col = warp_id % ${warps_per_block_n};
            
            // Shared memory for tiles
            __shared__ ${data_type} tileA[2][${block_size_m}][${block_size_k}];
            __shared__ ${data_type} tileB[2][${block_size_k}][${block_size_n}];
            
            // Tensor core fragment declarations
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, ${data_type}, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, ${data_type}, nvcuda::wmma::row_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, ${data_type}> c_frag;
            
            // Initialize accumulator fragment
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);
            
            // Starting positions
            register int a_row_start = block_row * ${block_size_m};
            register int b_col_start = block_col * ${block_size_n};
            
            // Double buffering index
            register int buffer_idx = 0;
            
            // Prefetch first tiles
            #pragma unroll 4
            for (int i = thread_id; i < ${block_size_m} * ${block_size_k}; i += blockDim.x * blockDim.y) {
                register int row = i / ${block_size_k};
                register int col = i % ${block_size_k};
                register int a_row = a_row_start + row;
                if (a_row < M && col < K) {
                    tileA[buffer_idx][row][col] = A[a_row * K + col];
                } else {
                    tileA[buffer_idx][row][col] = 0;
                }
            }
            
            #pragma unroll 4
            for (int i = thread_id; i < ${block_size_k} * ${block_size_n}; i += blockDim.x * blockDim.y) {
                register int row = i / ${block_size_n};
                register int col = i % ${block_size_n};
                register int b_col = b_col_start + col;
                if (row < K && b_col < N) {
                    tileB[buffer_idx][row][col] = B[row * N + b_col];
                } else {
                    tileB[buffer_idx][row][col] = 0;
                }
            }
            
            __syncthreads();
            
            // Main loop over K dimension
            for (int k_tile = 0; k_tile < (K + ${block_size_k} - 1) / ${block_size_k}; ++k_tile) {
                // Next buffer index for double buffering
                register int next_buffer_idx = 1 - buffer_idx;
                
                // Prefetch next tiles if not the last iteration
                if (k_tile < (K + ${block_size_k} - 1) / ${block_size_k} - 1) {
                    register int next_k_offset = (k_tile + 1) * ${block_size_k};
                    
                    #pragma unroll 4
                    for (int i = thread_id; i < ${block_size_m} * ${block_size_k}; i += blockDim.x * blockDim.y) {
                        register int row = i / ${block_size_k};
                        register int col = i % ${block_size_k};
                        register int a_row = a_row_start + row;
                        register int a_col = next_k_offset + col;
                        if (a_row < M && a_col < K) {
                            tileA[next_buffer_idx][row][col] = A[a_row * K + a_col];
                        } else {
                            tileA[next_buffer_idx][row][col] = 0;
                        }
                    }
                    
                    #pragma unroll 4
                    for (int i = thread_id; i < ${block_size_k} * ${block_size_n}; i += blockDim.x * blockDim.y) {
                        register int row = i / ${block_size_n};
                        register int col = i % ${block_size_n};
                        register int b_row = next_k_offset + row;
                        register int b_col = b_col_start + col;
                        if (b_row < K && b_col < N) {
                            tileB[next_buffer_idx][row][col] = B[b_row * N + b_col];
                        } else {
                            tileB[next_buffer_idx][row][col] = 0;
                        }
                    }
                }
                
                // Compute matrix multiplication for this tile using tensor cores
                // Process 16x16 blocks within the tile
                for (int m = 0; m < ${block_size_m}; m += 16) {
                    for (int n = 0; n < ${block_size_n}; n += 16) {
                        for (int k = 0; k < ${block_size_k}; k += 16) {
                            // Load fragments from shared memory
                            nvcuda::wmma::load_matrix_sync(a_frag, &tileA[buffer_idx][m][k], ${block_size_k});
                            nvcuda::wmma::load_matrix_sync(b_frag, &tileB[buffer_idx][k][n], ${block_size_n});
                            
                            // Perform matrix multiplication
                            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                }
                
                // Switch buffers
                buffer_idx = next_buffer_idx;
                
                // Synchronize before loading next tiles
                __syncthreads();
            }
            
            // Write results to global memory
            for (int m = 0; m < ${block_size_m}; m += 16) {
                for (int n = 0; n < ${block_size_n}; n += 16) {
                    // Calculate output position
                    register int c_row = a_row_start + m;
                    register int c_col = b_col_start + n;
                    
                    // Store fragment to global memory
                    if (c_row < M && c_col < N) {
                        ${data_type} c_tile[16][16];
                        nvcuda::wmma::store_matrix_sync(&c_tile[0][0], c_frag, 16, nvcuda::wmma::mem_row_major);
                        
                        // Apply alpha and beta
                        for (int i = 0; i < 16; ++i) {
                            for (int j = 0; j < 16; ++j) {
                                if (c_row + i < M && c_col + j < N) {
                                    int c_idx = (c_row + i) * N + (c_col + j);
                                    if (beta == 0) {
                                        C[c_idx] = alpha * c_tile[i][j];
                                    } else {
                                        C[c_idx] = alpha * c_tile[i][j] + beta * C[c_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
)";

// Helper function to get the GEMM kernel code
std::string get_gemm_kernel_code(DataType dtype, int block_size_m, int block_size_n, int block_size_k, bool use_unaligned_blocks = false, bool use_tensor_cores = false) {
    std::string data_type;
    switch (dtype) {
        case DataType::FLOAT32:
            data_type = "float";
            break;
        case DataType::FLOAT16:
            data_type = "half";
            break;
        default:
            throw std::runtime_error("Unsupported data type for GEMM");
    }
    
    // Calculate warp sizes and counts
    int warp_size = 32;
    
    // Tune block sizes for better rasterization
    if (use_unaligned_blocks) {
        // Use unaligned block sizes for better performance with non-power-of-2 matrices
        // These sizes are chosen to maximize SM occupancy and minimize padding
        
        // Define a set of efficient unaligned block sizes
        const std::vector<int> efficient_block_sizes = {
            32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256
        };
        
        // Find the most efficient block size for M dimension
        int best_block_size_m = 32;  // Default minimum
        int min_waste_m = block_size_m;  // Initialize with worst case
        
        for (int size : efficient_block_sizes) {
            if (size >= 32 && size <= block_size_m) {  // Ensure minimum size and not larger than original
                // Calculate waste (padding) for this block size
                int waste = (size - (block_size_m % size)) % size;
                
                // If this block size results in less waste, use it
                if (waste < min_waste_m) {
                    min_waste_m = waste;
                    best_block_size_m = size;
                }
            }
        }
        
        // Find the most efficient block size for N dimension
        int best_block_size_n = 32;  // Default minimum
        int min_waste_n = block_size_n;  // Initialize with worst case
        
        for (int size : efficient_block_sizes) {
            if (size >= 32 && size <= block_size_n) {  // Ensure minimum size and not larger than original
                // Calculate waste (padding) for this block size
                int waste = (size - (block_size_n % size)) % size;
                
                // If this block size results in less waste, use it
                if (waste < min_waste_n) {
                    min_waste_n = waste;
                    best_block_size_n = size;
                }
            }
        }
        
        // Update block sizes
        block_size_m = best_block_size_m;
        block_size_n = best_block_size_n;
    }
    else {
        // For non-unaligned blocks, ensure block sizes are multiples of warp size
        block_size_m = (block_size_m / warp_size) * warp_size;
        block_size_n = (block_size_n / warp_size) * warp_size;
    }
    
    // Adjust K dimension for better memory access
    block_size_k = (block_size_k / 8) * 8; // Make K a multiple of 8 for better memory access
    
    // Ensure minimum block sizes
    block_size_m = std::max(block_size_m, 32);
    block_size_n = std::max(block_size_n, 32);
    block_size_k = std::max(block_size_k, 8);
    
    int warps_per_block_m = block_size_m / warp_size;
    int warps_per_block_n = block_size_n / warp_size;
    
    // Handle non-power-of-2 block sizes
    int warp_size_m, warp_size_n;
    if (use_unaligned_blocks) {
        // For unaligned blocks, we need to calculate warp sizes differently
        // to ensure proper thread assignment
        warp_size_m = (block_size_m + warps_per_block_m - 1) / warps_per_block_m;
        warp_size_n = (block_size_n + warps_per_block_n - 1) / warps_per_block_n;
    } else {
        // For aligned blocks, use the standard calculation
        warp_size_m = warp_size / warps_per_block_n;
        warp_size_n = warp_size / warps_per_block_m;
    }
    
    // Calculate block threads and min blocks per SM
    int block_threads = warps_per_block_m * warps_per_block_n * warp_size;
    
    // Adjust min_blocks_per_sm based on matrix size and block size
    // For larger matrices, we can use more registers per thread by reducing min_blocks_per_sm
    // For smaller matrices, we want higher occupancy with more blocks per SM
    int min_blocks_per_sm;
    if (block_size_m >= 128 && block_size_n >= 128) {
        // For large blocks, use fewer blocks per SM to allow more registers per thread
        min_blocks_per_sm = 1;
    } else if (block_size_m >= 64 && block_size_n >= 64) {
        // For medium blocks, use a moderate number of blocks per SM
        min_blocks_per_sm = 2;
    } else {
        // For small blocks, maximize occupancy with more blocks per SM
        min_blocks_per_sm = 4;
    }
    
    // Choose the appropriate kernel template
    std::string code;
    if (use_tensor_cores) {
        code = GEMM_TENSOR_CORE_TEMPLATE;
    } else {
        code = GEMM_KERNEL_TEMPLATE;
    }
    
    // Replace template parameters
    code = std::regex_replace(code, std::regex("\\$\\{data_type\\}"), data_type);
    code = std::regex_replace(code, std::regex("\\$\\{block_size_m\\}"), std::to_string(block_size_m));
    code = std::regex_replace(code, std::regex("\\$\\{block_size_n\\}"), std::to_string(block_size_n));
    code = std::regex_replace(code, std::regex("\\$\\{block_size_k\\}"), std::to_string(block_size_k));
    code = std::regex_replace(code, std::regex("\\$\\{warps_per_block_m\\}"), std::to_string(warps_per_block_m));
    code = std::regex_replace(code, std::regex("\\$\\{warps_per_block_n\\}"), std::to_string(warps_per_block_n));
    code = std::regex_replace(code, std::regex("\\$\\{warp_size_m\\}"), std::to_string(warp_size_m));
    code = std::regex_replace(code, std::regex("\\$\\{warp_size_n\\}"), std::to_string(warp_size_n));
    code = std::regex_replace(code, std::regex("\\$\\{block_threads\\}"), std::to_string(block_threads));
    code = std::regex_replace(code, std::regex("\\$\\{min_blocks_per_sm\\}"), std::to_string(min_blocks_per_sm));
    code = std::regex_replace(code, std::regex("WARP_SIZE"), std::to_string(warp_size));
    
    return code;
}

// GEMM implementation
void gemm(const void* A, const void* B, void* C,
          int M, int N, int K,
          DataType dtype,
          float alpha, float beta,
          cudaStream_t stream) {
    
    // Use JIT compilation for the GEMM kernel
    if (dtype == DataType::FLOAT32) {
        // Get the kernel code
        int block_size_m = 128;
        int block_size_n = 128;
        int block_size_k = 32;
        
        // Adjust block sizes based on matrix dimensions for better performance
        if (M <= 64 || N <= 64) {
            // For small matrices, use smaller blocks
            block_size_m = 64;
            block_size_n = 64;
            block_size_k = 16;
        } else if (M >= 4096 || N >= 4096) {
            // For very large matrices, use larger blocks
            block_size_m = 256;
            block_size_n = 128;
            block_size_k = 32;
        }
        
        // Determine if we should use unaligned block sizes
        bool use_unaligned_blocks = false;
        
        // Check if matrix dimensions would benefit from unaligned block sizes
        // Non-power-of-2 dimensions often benefit from unaligned block sizes
        
        // Check if dimensions are not powers of 2
        bool is_m_power_of_2 = (M & (M - 1)) == 0;
        bool is_n_power_of_2 = (N & (N - 1)) == 0;
        
        // If either dimension is not a power of 2, consider using unaligned blocks
        if (!is_m_power_of_2 || !is_n_power_of_2) {
            // Calculate waste for standard block sizes
            int waste_m_128 = (128 - (M % 128)) % 128;
            int waste_n_128 = (128 - (N % 128)) % 128;
            
            // Calculate waste for unaligned block sizes
            int waste_m_112 = (112 - (M % 112)) % 112;
            int waste_n_112 = (112 - (N % 112)) % 112;
            int waste_m_96 = (96 - (M % 96)) % 96;
            int waste_n_96 = (96 - (N % 96)) % 96;
            
            // Use unaligned blocks if they result in less waste
            if (waste_m_112 < waste_m_128 || waste_n_112 < waste_n_128 ||
                waste_m_96 < waste_m_128 || waste_n_96 < waste_n_128) {
                use_unaligned_blocks = true;
            }
        }
        
        // Determine if we should use tensor cores
        bool use_tensor_cores = false;
        
        // Check if the GPU supports tensor cores
        cudaDeviceProp props;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        
        // Tensor cores are available on Volta (SM 7.0) and above
        bool tensor_cores_available = props.major >= 7;
        
        // Tensor cores work best with specific matrix sizes (multiples of 16)
        bool suitable_for_tensor_cores = 
            (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0) && 
            (M >= 64) && (N >= 64) && (K >= 64);
        
        // Use tensor cores if available and suitable
        if (tensor_cores_available && suitable_for_tensor_cores) {
            use_tensor_cores = true;
            
            // Adjust block sizes for tensor cores (must be multiples of 16)
            block_size_m = ((block_size_m + 15) / 16) * 16;
            block_size_n = ((block_size_n + 15) / 16) * 16;
            block_size_k = ((block_size_k + 15) / 16) * 16;
        }
        
        std::string kernel_code = get_gemm_kernel_code(dtype, block_size_m, block_size_n, block_size_k, use_unaligned_blocks, use_tensor_cores);
        
        // Set up JIT compilation parameters
        JitCompileParams params;
        params.kernel_name = use_tensor_cores ? "gemm_tensor_core_kernel" : "gemm_kernel";
        params.use_fast_math = true;
        params.use_tensor_cores = use_tensor_cores;
        params.block_size_m = block_size_m;
        params.block_size_n = block_size_n;
        params.block_size_k = block_size_k;
        params.use_unaligned_blocks = use_unaligned_blocks;
        
        // Add defines for the kernel
        params.defines["data_type"] = "float";
        params.defines["block_size_m"] = std::to_string(block_size_m);
        params.defines["block_size_n"] = std::to_string(block_size_n);
        params.defines["block_size_k"] = std::to_string(block_size_k);
        params.defines["use_unaligned_blocks"] = use_unaligned_blocks ? "1" : "0";
        
        // Get or compile the kernel
        auto& jit_compiler = get_jit_compiler();
        auto kernel = jit_compiler.get_or_compile(kernel_code, params);
        
        // Set up kernel arguments
        void* args[] = {
            (void*)&A,
            (void*)&B,
            (void*)&C,
            (void*)&M,
            (void*)&N,
            (void*)&K,
            (void*)&alpha,
            (void*)&beta
        };
        
        // Calculate grid and block dimensions
        int warps_per_block_m = block_size_m / 32;
        int warps_per_block_n = block_size_n / 32;
        int threads_per_block = warps_per_block_m * warps_per_block_n * 32;
        
        dim3 block(32, threads_per_block / 32);
        
        // Calculate optimal grid dimensions based on matrix size and GPU characteristics
        // Get device properties
        cudaDeviceProp props;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        
        // Calculate number of SMs and max blocks per SM
        int num_sms = props.multiProcessorCount;
        int max_blocks_per_sm = props.maxBlocksPerMultiProcessor;
        
        // Calculate grid dimensions to maximize SM utilization
        int grid_m = (M + block_size_m - 1) / block_size_m;
        int grid_n = (N + block_size_n - 1) / block_size_n;
        
        // Adjust grid dimensions to balance workload across SMs
        // We want to ensure that each SM gets a similar number of blocks
        int total_blocks = grid_m * grid_n;
        int target_blocks_per_sm = std::min(max_blocks_per_sm, (total_blocks + num_sms - 1) / num_sms);
        
        // Adjust grid dimensions if necessary to better balance workload
        if (total_blocks > num_sms * target_blocks_per_sm) {
            // If we have more blocks than SMs can handle efficiently,
            // adjust grid dimensions to reduce total blocks while maintaining coverage
            float aspect_ratio = static_cast<float>(grid_m) / grid_n;
            int new_grid_m = std::sqrt(num_sms * target_blocks_per_sm * aspect_ratio);
            int new_grid_n = (num_sms * target_blocks_per_sm) / new_grid_m;
            
            // Ensure we still cover the entire matrix
            new_grid_m = std::max(new_grid_m, grid_m);
            new_grid_n = std::max(new_grid_n, grid_n);
            
            grid_m = new_grid_m;
            grid_n = new_grid_n;
        }
        
        dim3 grid(grid_n, grid_m);
        
        // Launch the kernel
        CUstream cu_stream = (CUstream)stream;
        kernel->launch(cu_stream, args, grid, block);
    }
    else {
        throw std::runtime_error("Unsupported data type for GEMM");
    }
}

} // namespace cuda
} // namespace celeris 