# Tri-RMSNorm

This small package provides an custom GPU kernel for Root Mean Square layer normalization process with fused operations, leveraging the [Triton compiler by OpenAI](https://github.com/openai/triton) for high performance and parallel computations on GPUs. Implementation includes both forward and backward passes of RMS layer normalization, optimized for specifically empowering deep learning training and inferencing.

## Features

**Customized FW/BW RMS Normalization:** 

- Implements the forward and backward passes of RMS normalization with fused operations for better performance.

**Triton and PyTorch Integration:** 

- Utilizes Triton for GPU-accelerated computations and parallel computation, seamlessly integrated with PyTorch tensors.

**Customizable:**

- Compile-time constants for block sizes, accommodating different GPU architectures and memory layouts.

**Atomic Operations for Gradient Accumulation:** 

- Atomic operations to safely accumulate gradients across threads, preventing race conditions and ensuring correct gradient computation during the backward pass.

**Lock-Free Mechanisms:** 

- Advanced sync to minimize locking and blocking, improving the performance and scalability of gradient computation.

## Getting Started

**Requirements**

```bash
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchvision==0.16.0+cu121
triton==2.1.0
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/attophyd/Tri-RMSNorm
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
    input = torch.randn((1024, 1024), device='cuda', requires_grad=True)
    biases = torch.randn(1024, device='cuda', requires_grad=True)
    weights = torch.randn(1024, device='cuda', requires_grad=True)

    output = RMSNormFunctionCustomKernel.apply(input, weights, biases, eps)
    loss = output.mean()
    loss.backward()

    print("Grad X: ", input.grad)
    print("Grad W: ", weights.grad)
    print("Grad B: ", biases.grad)

test_rms_norm_custom_kernel()
```

Adjust grid, block, and other parameters as per your requirements and GPU specifications.

## Benchmark

Tri-RMSNorm kernel demonstrates improved speedup in initial benchmarks when compared to the PyTorch-based custom RMSNorm implementation. Benchmarks will be included in the repository to ensure reproducibility. Compared to the LayerNorm custom kernel and the Tri-RMSNorm kernel, considering both the forward and backward passes, the mean speedup is approximately 28.57%. This respects the original results' range introduced in the RMSNorm paper, which states it "reduces the running time by 7% to 64% on different models." When compared to the standalone PyTorch RMSNorm implementation, and the Tri-RMSNorm kernel, considering both the forward and backward passes computations, yields a mean speedup of approximately 10.18%. For GB/s comparisons for both implementation, analyzed Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.

Please note that:

- All benchmark tests will be released in the repository soon.
- All tests are conducted on a NVIDIA T4 Tensor Core GPU, will be reproduced with A100 and 4090.
- Model training wasn't performed, with the customized kernel.
- A comparison with custom fused RMSNorm CUDA kernels implemented in xFormers has not been conducted, as of yet.

## License

This package is licensed under the Apache License - see the LICENSE file for details.
