import unittest
import torch
from celeris.utils.gpu_utils import benchmark_matmul, get_gpu_info, get_optimal_dtype
from celeris import randn, add, mul, matmul, relu, sigmoid, tanh


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_matmul(self):
        """Test matrix multiplication benchmark for a 512x512 matrix."""
        result = benchmark_matmul(size=512, iterations=5)
        # Ensure required keys are in the result
        self.assertIn('time_seconds', result)
        self.assertIn('gflops', result)
        print("Matrix multiplication benchmark (512x512):", result)

    def test_gpu_info(self):
        """Test that GPU info is returned correctly."""
        gpu_info = get_gpu_info()
        if torch.cuda.is_available():
            self.assertGreater(len(gpu_info), 0)
        else:
            self.assertEqual(len(gpu_info), 0)
        print("GPU info:", gpu_info)

    def test_algebra_operations(self):
        """Test element-wise add, multiply, matrix multiplication and activations."""
        x = randn(10, 10)
        y = randn(10, 10)
        sum_result = add(x, y)
        self.assertEqual(sum_result.size(), x.size())
        prod_result = mul(x, y)
        self.assertEqual(prod_result.size(), x.size())
        mm_result = matmul(x, y)
        self.assertEqual(mm_result.size(), (10, 10))
        relu_result = relu(x)
        self.assertEqual(relu_result.size(), x.size())
        sigmoid_result = sigmoid(x)
        self.assertEqual(sigmoid_result.size(), x.size())
        tanh_result = tanh(x)
        self.assertEqual(tanh_result.size(), x.size())
        print("Algebra operations test passed.")

    def test_optimal_dtype(self):
        """Test that optimal data type is one of the expected types."""
        optimal_dtype = get_optimal_dtype()
        self.assertIn(optimal_dtype, [torch.float32, torch.float16, torch.bfloat16])
        print("Optimal dtype:", optimal_dtype)


if __name__ == '__main__':
    unittest.main() 