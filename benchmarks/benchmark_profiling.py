import os
import sys
import time
import io
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pstats

# Add the parent directory to import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celeris
from celeris.config import get_config, set_config
from celeris.utils import print_gpu_summary


def benchmark_operation(op_name, op_func, shape, num_runs=10):
    """
    Benchmark a given operation for a specified shape.
    Returns average elapsed time per run and throughput in GOPS.
    """
    # Prepare tensors based on operation type
    if op_name == 'MatMul':
        np_data1 = np.random.randn(shape[0], shape[1]).astype(np.float32)
        np_data2 = np.random.randn(shape[1], shape[0]).astype(np.float32)
        t1 = celeris.from_numpy(np_data1)
        t2 = celeris.from_numpy(np_data2)
    else:
        np_data1 = np.random.randn(*shape).astype(np.float32)
        np_data2 = np.random.randn(*shape).astype(np.float32)
        t1 = celeris.from_numpy(np_data1)
        t2 = celeris.from_numpy(np_data2)

    # Warm-up runs
    for _ in range(2):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            _ = op_func(t1, t2)
        else:
            _ = op_func(t1)
    
    start = time.time()
    for _ in range(num_runs):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            _ = op_func(t1, t2)
        else:
            _ = op_func(t1)
    elapsed = (time.time() - start) / num_runs
    
    # Calculate number of elements processed
    elements = np.prod(shape)
    throughput = elements / elapsed / 1e9  # GOPS
    return elapsed, throughput


def run_profiling():
    """
    Run benchmarks for various operations on different matrix sizes.
    Returns a nested dictionary with operation names as keys and shape tuples mapped to (elapsed, throughput).
    """
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    operations = [
        ('Element-wise Add', celeris.add),
        ('Element-wise Mul', celeris.mul),
        ('MatMul', celeris.matmul),
        ('ReLU', celeris.relu),
        ('Sigmoid', celeris.sigmoid),
        ('Tanh', celeris.tanh)
    ]
    data = {}
    for op_name, op_func in operations:
        data[op_name] = {}
        for shape in sizes:
            elapsed, throughput = benchmark_operation(op_name, op_func, shape, num_runs=10)
            data[op_name][shape] = (elapsed, throughput)
            print(f"{op_name} with shape {shape}: {elapsed:.6f} s, {throughput:.2f} GOPS")
    return data


def plot_performance(data):
    """
    Generate line plots for each operation showing throughput vs matrix size.
    Plots are saved to the 'docs/images' directory.
    """
    for op_name, results in data.items():
        sizes = []
        gflops_values = []
        for shape, (elapsed, throughput) in results.items():
            # For square matrices, use one dimension
            sizes.append(shape[0])
            gflops_values.append(throughput)
        plt.figure()
        plt.plot(sizes, gflops_values, marker='o')
        plt.xlabel("Matrix Dimension (n for n x n)")
        plt.ylabel("Throughput (GOPS)")
        plt.title(f"Performance of {op_name}")
        plt.grid(True)
        filename = os.path.join("docs", "images", f"{op_name.replace(' ', '_').lower()}_performance.png")
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")


def run_cprofile_analysis():
    """
    Profile a heavy matrix multiplication operation using cProfile and save the stats.
    """
    pr = cProfile.Profile()
    size = 1024
    np_data1 = np.random.randn(size, size).astype(np.float32)
    np_data2 = np.random.randn(size, size).astype(np.float32)
    t1 = celeris.from_numpy(np_data1)
    t2 = celeris.from_numpy(np_data2)
    print("Profiling matrix multiplication (1024x1024)...")
    pr.enable()
    _ = celeris.matmul(t1, t2)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    profile_output = s.getvalue()
    profile_file = os.path.join("docs", "images", "matmul_cprofile.txt")
    with open(profile_file, "w") as f:
        f.write(profile_output)
    print(f"CProfile stats saved to {profile_file}")


def main():
    # Ensure docs/images directory exists
    os.makedirs(os.path.join("docs", "images"), exist_ok=True)

    print("=== Celeris GPU Summary ===")
    print_gpu_summary()
    
    print("\n=== Running In-Depth Profiling and Benchmarking ===")
    data = run_profiling()
    
    # Save benchmark data as a markdown table
    table = []
    for op_name, results in data.items():
        for shape, (elapsed, throughput) in results.items():
            table.append([op_name, f"{shape[0]}x{shape[1]}", f"{elapsed:.6f}", f"{throughput:.2f}"])
    headers = ["Operation", "Shape", "Avg Time (s)", "Throughput (GOPS)"]
    table_md = tabulate(table, headers=headers, tablefmt="github")
    profile_results_file = os.path.join("docs", "images", "benchmark_profiling.md")
    with open(profile_results_file, "w") as f:
        f.write(table_md)
    print(f"Benchmark profiling results saved to {profile_results_file}")
    
    # Generate plots for each operation
    plot_performance(data)
    
    # Run cProfile analysis on a heavy operation
    run_cprofile_analysis()
    
    print("\nIn-depth profiling and benchmarking complete.")


if __name__ == '__main__':
    main() 