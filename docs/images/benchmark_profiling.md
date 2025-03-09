| Operation        | Shape     |   Avg Time (s) |   Throughput (GOPS) |
|------------------|-----------|----------------|---------------------|
| Element-wise Add | 512x512   |          5e-06 |               50.44 |
| Element-wise Add | 1024x1024 |          5e-06 |              232.7  |
| Element-wise Add | 2048x2048 |          5e-06 |              925.9  |
| Element-wise Mul | 512x512   |          5e-06 |               54.43 |
| Element-wise Mul | 1024x1024 |          5e-06 |              230.26 |
| Element-wise Mul | 2048x2048 |          5e-06 |              911.51 |
| MatMul           | 512x512   |          7e-06 |               35.81 |
| MatMul           | 1024x1024 |          8e-06 |              128.6  |
| MatMul           | 2048x2048 |          8e-06 |              523.58 |
| ReLU             | 512x512   |          5e-06 |               50.44 |
| ReLU             | 1024x1024 |          5e-06 |              217.73 |
| ReLU             | 2048x2048 |          5e-06 |              875.23 |
| Sigmoid          | 512x512   |          5e-06 |               55.81 |
| Sigmoid          | 1024x1024 |          5e-06 |              232.7  |
| Sigmoid          | 2048x2048 |          4e-06 |              945.82 |
| Tanh             | 512x512   |          5e-06 |               55.53 |
| Tanh             | 1024x1024 |          4e-06 |              239.02 |
| Tanh             | 2048x2048 |          5e-06 |              925.9  |