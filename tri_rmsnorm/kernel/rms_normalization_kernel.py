import torch
import triton
import triton.language as tl

if hasattr(tl, "libdevice"):
    tl_math = tl.libdevice
else:
    tl_math = tl.math


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel invocation for forward pass of RMS normalization with fused operations

    Params:
        - X (tensor): Input tensor
        - Y (tensor): Output tensor where the normalized results will be written
        - W (tensor): Scale tensor applied to the normalized input
        - B (tensor): Bias tensor added to the scaled input
        - Rstd (tensor): Reciprocal of the standard deviation used for normalization
        - stride (int): Stride to be applied when accessing elements in the input and output tensors
        - N (int): Number of elements in the input tensor
        - eps (float): Small epsilon value added to the variance to prevent division by zero
        - BLOCK_SIZE (constexpr): Size of the block for computation, provided as a compile-time constant

    Return:
        - None

    Usage:
        _rms_norm_fwd_fused[grid, block](X, Y, W, B, Rstd, stride, N, eps, BLOCK_SIZE)
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    _rms = 0
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    tl.store(Rstd + row, rms)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_hat = x / rms
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,
    DY,
    DW,
    DB,
    X,
    W,
    B,
    Rstd,
    Lock,
    stride,
    N,
    eps,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel invocation for backward pass of RMS normalization, computing gradients w.r.t. input

    Params:
        - DX (tensor): Gradient of the loss with respect to the inputs
        - DY (tensor): Gradient of the loss with respect to the outputs
        - DW (tensor): Gradient of the loss with respect to the scale tensor W
        - DB (tensor): Gradient of the loss with respect to the bias tensor B
        - X (tensor): Input tensor from the forward pass
        - W (tensor): Scale tensor applied during the forward pass
        - B (tensor): Bias tensor added during the forward pass
        - Rstd (tensor): Reciprocal of the standard deviation used for normalization in the forward pass
        - Lock (tensor): Lock tensor for atomic operations to prevent race conditions
        - stride (int): Stride to be applied when accessing elements in the tensors
        - N (int): Number of elements in each tensor
        - eps (float): Small epsilon value used during the forward pass
        - GROUP_SIZE_M (constexpr): Size of the group for M dimension, provided as a compile-time constant
        - BLOCK_SIZE_N (constexpr): Size of the block for N dimension, provided as a compile-time constant

    Return:
        - None

    Usage:
        _rms_norm_bwd_dx_fused[grid, block](DX, DY, DW, DB, X, W, B, Rstd, Lock, stride, N, eps, GROUP_SIZE_M, BLOCK_SIZE_N)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)
    x_norm = x * rstd
    wdy = w * dy
    dx = wdy * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * x_norm).to(w.dtype)
    partial_db = dy.to(w.dtype)

    # Locking mechanism to prevent race conditions
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _rms_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    Kernel invocation for backward pass of RMS normalization, computing and aggregating gradients w.r.t. weights and biases

    Params:
        - DW (tensor): Intermediate gradient tensor for the scale factors, W
        - DB (tensor): Intermediate gradient tensor for the biases, B
        - FINAL_DW (tensor): Aggregated gradient tensor for the scale factors, to be updated
        - FINAL_DB (tensor): Aggregated gradient tensor for the biases, to be updated
        - M (int): Number of groups or batch size dimension
        - N (int): Dimensionality of the feature vectors or the number of features
        - BLOCK_SIZE_M (constexpr): Compile-time constant defining the block size in the M dimension
        - BLOCK_SIZE_N (constexpr): Compile-time constant defining the block size in the N dimension

    Return:
        - None

    Usage:
        _rms_norm_bwd_dwdb[grid, block](DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
