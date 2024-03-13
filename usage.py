import torch
from tri_rmsnorm.kernel.rms_normalization_kernel import (
    _rms_norm_fwd_fused,
    _rms_norm_bwd_dx_fused,
)


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
        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            _db,
            x,
            weight,
            bias,
            rstd,
            locks,
            x.stride(0),
            N,
            eps,
            GROUP_SIZE_M=32,
            BLOCK_SIZE_N=1024,
        )
        return dx, _dw, _db, None


def test_rms_norm_custom_kernel():
    eps = 1e-5
    input = torch.tensor([[0.1, -0.2] * 10] * 10, device="cuda", requires_grad=True)
    weights = torch.tensor([0.1] * 20, device="cuda", requires_grad=True)
    biases = torch.tensor([0.01] * 20, device="cuda", requires_grad=True)

    output = RMSNormFunctionCustomKernel.apply(input, weights, biases, eps)
    loss = output.mean()
    loss.backward()

    print("Grads X: ", input.grad)
    print("Grads W: ", weights.grad)
    print("Grads B: ", biases.grad)


test_rms_norm_custom_kernel()
