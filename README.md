# Tri-RMSNorm

This small package provides an custom Triton kernel of RMS layer normalization with fused operations, leveraging the Triton compiler by OpenAI for high performance on GPUs. Implementation includes both forward and backward passes of RMS layer normalization, optimized for empowering deep learning training and inferencing.

## Features

**Customized FW/BW RMS Normalization:** 

Implements the forward and backward passes of RMS normalization with fused operations for better performance.

**Triton and PyTorch Integration:** 

Utilizes Triton for GPU-accelerated computations and parallel computation, seamlessly integrated with PyTorch tensors.

**Customizable:**

Compile-time constants for block sizes, accommodating different GPU architectures and memory layouts.

**Atomic Operations for Gradient Accumulation:** 

Atomic operations to safely accumulate gradients across threads, preventing race conditions and ensuring correct gradient computation during the backward pass.

**Lock-Free Mechanisms:** 

Advanced sync to minimize locking and blocking, improving the performance and scalability of gradient computation.

## Getting Started

## **Installation**

**Requirements**

```bash
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchvision==0.16.0+cu121
triton==2.1.0
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/simudt/Tri-RMSNorm
cd Tri-RMSNorm
pip3 install -e .
```

## Usage

The package provides two main functions:

- `_rms_norm_fwd_fused` for the forward pass of RMS normalization

- `_rms_norm_bwd_dx_fused` for the backward pass, computing gradients with respect to X, W, B

```python
class RMSNormFunctionCustomKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        _rms_norm_fwd_fused[(M,)](x, y, weight, bias, rstd, x.stride(0), N, eps, BLOCK_SIZE=1024)
        ctx.save_for_backward(x, weight, bias, rstd)
        ctx.eps = eps
        ctx.N = N
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, rstd = ctx.saved_tensors
        eps = ctx.eps
        N = ctx.N
        M = x.shape[0]
        dx = torch.empty_like(x)
        _dw = torch.empty_like(weight)
        _db = torch.empty_like(bias)
        locks = torch.zeros(2 * 32, dtype=torch.int32, device=x.device)
        _rms_norm_bwd_dx_fused[(M,)](dx, dy, _dw, _db, x, weight, bias, rstd, locks, x.stride(0), N, eps, GROUP_SIZE_M=32, BLOCK_SIZE_N=1024)
        return dx, _dw, _db, None

def test_rms_norm_custom_kernel():
    eps = 1e-5
    input = torch.tensor([[0.1, -0.2] * 10] * 10, device='cuda', requires_grad=True)
    weights = torch.tensor([0.1] * 20, device='cuda', requires_grad=True)
    biases = torch.tensor([0.01] * 20, device='cuda', requires_grad=True)

    output = RMSNormFunctionCustomKernel.apply(input, weights, biases, eps)
    loss = output.mean()
    loss.backward()

    print("Gradients on Input: ", input.grad)
    print("Gradients on Weights: ", weights.grad)
    print("Gradients on Biases: ", biases.grad)

test_rms_norm_custom_kernel()
```

Adjust grid, block, and other parameters as per your requirements and GPU specifications.

## Benchmark

Tri-RMSNorm kernel demonstrates improved speedup in initial benchmarks when compared to the PyTorch-based custom RMSNorm implementation. Benchmarks will be included in the repository to ensure reproducibility.

## License

This package is licensed under the Apache License - see the LICENSE file for details.