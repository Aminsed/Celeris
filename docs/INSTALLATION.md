# Installation Guide for Celeris

This guide will walk you through the steps to install and set up Celeris on your machine. You do not need to know C++ or CUDA; everything is built to work with Python!

## Prerequisites

1. **Python 3.6 or Later**: Make sure you have Python 3.6 or a newer version installed.
2. **Virtual Environment (Recommended)**: It is best to use a Python virtual environment so your setup remains clean and isolated.
3. **Git**: You need Git installed to clone the repository.
4. **CUDA (Optional)**: If you have an NVIDIA GPU and want to use GPU acceleration, ensure that CUDA is installed. If not, Celeris will run in CPU mode automatically.

## Step-by-Step Installation

### 1. Clone the Repository

Open your terminal and run the following commands:
```bash
git clone <repository_url>
cd deepgemm
```

### 2. Set Up a Virtual Environment

You can use Python's built-in `venv`. For example:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies and Celeris

With your virtual environment activated, install all required packages and the Celeris library in editable mode by running:
```bash
pip install -e .
```

### 4. Run the Test Suite (Optional)

To confirm that everything is working correctly, run the test suite:
```bash
python tests/run_all_tests.py
```

## Troubleshooting Tips

- **GPU Issues**: If you do not have an NVIDIA GPU or CUDA installed, Celeris will automatically run in CPU mode.
- **Virtual Environment Activation**: Ensure your virtual environment is activated (e.g., `source myenv/bin/activate`).
- **Dependency Errors**: If you encounter errors related to missing packages, try updating pip:
  ```bash
  pip install --upgrade pip
  ```
- **Need Help?**: Check the project issue tracker or contact the maintainer if you experience problems.

## Next Steps

Once installed, you can explore the example scripts in the `examples/` directory or run benchmarks to test Celeris' performance.

Happy computing! 