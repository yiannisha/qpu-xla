from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from math import lcm
from pathlib import Path
from time import CLOCK_MONOTONIC, clock_gettime
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:
    torch = None
    torch_f = None

from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu
from videocore7.driver import Array, Driver


def _load_example_module(stem: str, name: str) -> Any:
    module_path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TILEDCONV2D = _load_example_module("tiledconv2d", "_tiledlenet5_tiledconv2d")
_TILEDMLP = _load_example_module("tiledmlp", "_tiledlenet5_tiledmlp")

numpy_conv2d_nchw = _TILEDCONV2D.numpy_conv2d_nchw
qpu_tiledconv2d_fp32 = _TILEDCONV2D.qpu_tiledconv2d_fp32
qpu_tiledconv2d_int32 = _TILEDCONV2D.qpu_tiledconv2d_int32
reshape_weight_oihw_to_gemm = _TILEDCONV2D._reshape_weight_oihw_to_gemm
qpu_tiled_bias_activation_fp32 = _TILEDMLP.qpu_tiled_bias_activation_fp32
qpu_tiled_bias_activation_int32 = _TILEDMLP.qpu_tiled_bias_activation_int32

INPUT_CHANNELS = 1
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
KERNEL = 5
POOL_KERNEL = 2
POOL_STRIDE = 2
CONV1_OUT_CHANNELS = 6
CONV2_OUT_CHANNELS = 16
CONV1_OUT_HEIGHT = INPUT_HEIGHT - KERNEL + 1
CONV1_OUT_WIDTH = INPUT_WIDTH - KERNEL + 1
POOL1_OUT_HEIGHT = CONV1_OUT_HEIGHT // POOL_STRIDE
POOL1_OUT_WIDTH = CONV1_OUT_WIDTH // POOL_STRIDE
CONV2_OUT_HEIGHT = POOL1_OUT_HEIGHT - KERNEL + 1
CONV2_OUT_WIDTH = POOL1_OUT_WIDTH - KERNEL + 1
POOL2_OUT_HEIGHT = CONV2_OUT_HEIGHT // POOL_STRIDE
POOL2_OUT_WIDTH = CONV2_OUT_WIDTH // POOL_STRIDE
FC1_IN_FEATURES = CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH
FC1_OUT_FEATURES = 120
FC2_OUT_FEATURES = 84
FC3_OUT_FEATURES = 10
P_TILE = 16
Q_TILE = 4
R_TILE = 16
SIGNED_24BIT_LIMIT = 1 << 23


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def _round_up(value: int, tile: int) -> int:
    return ((value + tile - 1) // tile) * tile


def _benchmark_callable(
    fn: Callable[[], npt.NDArray[np.generic]],
    *,
    warmup: int = 1,
    repeat: int = 3,
) -> tuple[npt.NDArray[np.generic], float]:
    result = fn()
    for _ in range(warmup):
        result = fn()
    best_sec = float("inf")
    for _ in range(repeat):
        start = getsec()
        result = fn()
        best_sec = min(best_sec, getsec() - start)
    return result, best_sec


def _benchmark_timing(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeat: int = 5,
) -> float:
    fn()
    for _ in range(warmup):
        fn()
    best_sec = float("inf")
    for _ in range(repeat):
        start = getsec()
        fn()
        best_sec = min(best_sec, getsec() - start)
    return best_sec


def _max_abs_int(x: npt.NDArray[np.integer[Any]]) -> int:
    if x.size == 0:
        return 0
    x64 = x.astype(np.int64, copy=False)
    return int(np.max(np.abs(x64)))


def _ensure_int32_range(x: npt.NDArray[np.int64], *, what: str) -> None:
    max_abs = int(np.max(np.abs(x), initial=0))
    if max_abs > np.iinfo(np.int32).max:
        raise ValueError(f"{what} exceeds the signed int32 range")


def _checked_int32(x: npt.NDArray[np.int64], *, what: str) -> npt.NDArray[np.int32]:
    _ensure_int32_range(x, what=what)
    return np.ascontiguousarray(x.astype(np.int32))


def trunc_divide_pow2_numpy(x: npt.NDArray[np.int64], shift: int) -> npt.NDArray[np.int64]:
    bias = ((x < 0).astype(np.int64)) * ((1 << shift) - 1)
    return (x + bias) >> shift


def lenet5_ops(batch: int) -> int:
    conv1_macs = batch * CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH * INPUT_CHANNELS * KERNEL * KERNEL
    conv2_macs = (
        batch
        * CONV2_OUT_CHANNELS
        * CONV2_OUT_HEIGHT
        * CONV2_OUT_WIDTH
        * CONV1_OUT_CHANNELS
        * KERNEL
        * KERNEL
    )
    pool1_ops = batch * CONV1_OUT_CHANNELS * POOL1_OUT_HEIGHT * POOL1_OUT_WIDTH * 4
    pool2_ops = batch * CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH * 4
    fc1_ops = 2 * batch * FC1_IN_FEATURES * FC1_OUT_FEATURES + 2 * batch * FC1_OUT_FEATURES
    fc2_ops = 2 * batch * FC1_OUT_FEATURES * FC2_OUT_FEATURES + 2 * batch * FC2_OUT_FEATURES
    fc3_ops = 2 * batch * FC2_OUT_FEATURES * FC3_OUT_FEATURES + batch * FC3_OUT_FEATURES
    conv1_bias_relu = 2 * batch * CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH
    conv2_bias_relu = 2 * batch * CONV2_OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH
    return (
        2 * conv1_macs
        + 2 * conv2_macs
        + pool1_ops
        + pool2_ops
        + fc1_ops
        + fc2_ops
        + fc3_ops
        + conv1_bias_relu
        + conv2_bias_relu
    )


def lenet5_gops(batch: int, sec: float) -> float:
    return lenet5_ops(batch) / sec * 1e-9


@dataclass(frozen=True)
class LeNet5Weights:
    conv1_w: npt.NDArray[np.generic]
    conv1_b: npt.NDArray[np.generic]
    conv2_w: npt.NDArray[np.generic]
    conv2_b: npt.NDArray[np.generic]
    fc1_w: npt.NDArray[np.generic]
    fc1_b: npt.NDArray[np.generic]
    fc2_w: npt.NDArray[np.generic]
    fc2_b: npt.NDArray[np.generic]
    fc3_w: npt.NDArray[np.generic]
    fc3_b: npt.NDArray[np.generic]


@dataclass(frozen=True)
class LeNet5Reference:
    conv1: npt.NDArray[np.generic]
    pool1: npt.NDArray[np.generic]
    conv2: npt.NDArray[np.generic]
    pool2: npt.NDArray[np.generic]
    flat: npt.NDArray[np.generic]
    fc1: npt.NDArray[np.generic]
    fc2: npt.NDArray[np.generic]
    output: npt.NDArray[np.generic]


@dataclass(frozen=True)
class MatrixView:
    rows: int
    cols: int
    row_stride: int
    base_addr: int
    itemsize: int


@dataclass(frozen=True)
class Dispatch:
    code: Array[np.uint64]
    uniforms: Array[np.uint32]
    thread: int
    workgroup: tuple[int, int, int] | None = None
    wgs_per_sg: int = 16


@dataclass(frozen=True)
class QpuBenchmarkStats:
    prep_sec: float
    cached_total_sec: float
    execute_only_sec: float
    max_abs_error: float


@dataclass(frozen=True)
class StageTiming:
    label: str
    sec: float


@dataclass(frozen=True)
class DTypeConfig:
    name: str
    np_dtype: np.dtype[Any]
    matmul_kernel: Callable[[Assembly], None]
    bias_kernel: Callable[..., None]
    pool_kernel: Callable[..., None]


def numpy_avgpool2d_fp32(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    return np.ascontiguousarray(((x00 + x01) + (x10 + x11)) * np.float32(0.25), dtype=np.float32)


def numpy_avgpool2d_int64(x: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    acc = x00 + x01 + x10 + x11
    return np.ascontiguousarray(trunc_divide_pow2_numpy(acc, 2), dtype=np.int64)


def reference_lenet5_fp32(
    x: npt.NDArray[np.float32],
    weights: LeNet5Weights,
) -> LeNet5Reference:
    conv1 = numpy_conv2d_nchw(
        x,
        weights.conv1_w,
        compute_dtype=np.float32,
        out_dtype=np.float32,
    )
    conv1 = np.ascontiguousarray(
        np.maximum(conv1 + weights.conv1_b.reshape(1, -1, 1, 1), np.float32(0.0)),
        dtype=np.float32,
    )
    pool1 = numpy_avgpool2d_fp32(conv1)

    conv2 = numpy_conv2d_nchw(
        pool1,
        weights.conv2_w,
        compute_dtype=np.float32,
        out_dtype=np.float32,
    )
    conv2 = np.ascontiguousarray(
        np.maximum(conv2 + weights.conv2_b.reshape(1, -1, 1, 1), np.float32(0.0)),
        dtype=np.float32,
    )
    pool2 = numpy_avgpool2d_fp32(conv2)

    flat = np.ascontiguousarray(pool2.transpose(0, 2, 3, 1).reshape(x.shape[0], FC1_IN_FEATURES), dtype=np.float32)
    fc1 = np.ascontiguousarray(np.maximum(flat.dot(weights.fc1_w) + weights.fc1_b, np.float32(0.0)), dtype=np.float32)
    fc2 = np.ascontiguousarray(np.maximum(fc1.dot(weights.fc2_w) + weights.fc2_b, np.float32(0.0)), dtype=np.float32)
    output = np.ascontiguousarray(fc2.dot(weights.fc3_w) + weights.fc3_b, dtype=np.float32)
    return LeNet5Reference(
        conv1=conv1,
        pool1=pool1,
        conv2=conv2,
        pool2=pool2,
        flat=flat,
        fc1=fc1,
        fc2=fc2,
        output=output,
    )


def reference_lenet5_int32(
    x: npt.NDArray[np.int32],
    weights: LeNet5Weights,
) -> LeNet5Reference:
    conv1 = numpy_conv2d_nchw(
        x,
        weights.conv1_w,
        compute_dtype=np.int64,
        out_dtype=np.int64,
    )
    conv1 = _checked_int32(
        np.maximum(conv1 + weights.conv1_b.astype(np.int64).reshape(1, -1, 1, 1), 0),
        what="conv1 activation",
    )
    pool1 = _checked_int32(numpy_avgpool2d_int64(conv1.astype(np.int64, copy=False)), what="pool1 activation")

    conv2 = numpy_conv2d_nchw(
        pool1,
        weights.conv2_w,
        compute_dtype=np.int64,
        out_dtype=np.int64,
    )
    conv2 = _checked_int32(
        np.maximum(conv2 + weights.conv2_b.astype(np.int64).reshape(1, -1, 1, 1), 0),
        what="conv2 activation",
    )
    pool2 = _checked_int32(numpy_avgpool2d_int64(conv2.astype(np.int64, copy=False)), what="pool2 activation")

    flat64 = np.ascontiguousarray(
        pool2.transpose(0, 2, 3, 1).reshape(x.shape[0], FC1_IN_FEATURES).astype(np.int64, copy=False),
        dtype=np.int64,
    )
    fc1 = _checked_int32(
        np.maximum(flat64.dot(weights.fc1_w.astype(np.int64)) + weights.fc1_b.astype(np.int64), 0),
        what="fc1 activation",
    )
    fc2 = _checked_int32(
        np.maximum(fc1.astype(np.int64).dot(weights.fc2_w.astype(np.int64)) + weights.fc2_b.astype(np.int64), 0),
        what="fc2 activation",
    )
    output = _checked_int32(
        fc2.astype(np.int64).dot(weights.fc3_w.astype(np.int64)) + weights.fc3_b.astype(np.int64),
        what="output activation",
    )
    flat = np.ascontiguousarray(flat64.astype(np.int32))
    return LeNet5Reference(
        conv1=conv1,
        pool1=pool1,
        conv2=conv2,
        pool2=pool2,
        flat=flat,
        fc1=fc1,
        fc2=fc2,
        output=output,
    )


def numpy_lenet5_fp32(x: npt.NDArray[np.float32], weights: LeNet5Weights) -> npt.NDArray[np.float32]:
    return reference_lenet5_fp32(x, weights).output


def numpy_lenet5_int32(x: npt.NDArray[np.int32], weights: LeNet5Weights) -> npt.NDArray[np.int32]:
    return reference_lenet5_int32(x, weights).output


def torch_avgpool2d_int32(x_t: Any) -> Any:
    x00 = x_t[:, :, 0::2, 0::2]
    x01 = x_t[:, :, 0::2, 1::2]
    x10 = x_t[:, :, 1::2, 0::2]
    x11 = x_t[:, :, 1::2, 1::2]
    acc = x00.to(torch.int64) + x01.to(torch.int64) + x10.to(torch.int64) + x11.to(torch.int64)
    return torch.div(acc, 4, rounding_mode="trunc").to(torch.int32)


def make_torch_runner_fp32(
    x: npt.NDArray[np.float32],
    weights: LeNet5Weights,
) -> Callable[[], Any]:
    if torch is None or torch_f is None:
        raise RuntimeError("torch is not available")

    x_t = torch.from_numpy(np.ascontiguousarray(x))
    conv1_w_t = torch.from_numpy(np.ascontiguousarray(weights.conv1_w))
    conv1_b_t = torch.from_numpy(np.ascontiguousarray(weights.conv1_b)).view(1, -1, 1, 1)
    conv2_w_t = torch.from_numpy(np.ascontiguousarray(weights.conv2_w))
    conv2_b_t = torch.from_numpy(np.ascontiguousarray(weights.conv2_b)).view(1, -1, 1, 1)
    fc1_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc1_w))
    fc1_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc1_b))
    fc2_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc2_w))
    fc2_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc2_b))
    fc3_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc3_w))
    fc3_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc3_b))

    def run() -> Any:
        with torch.no_grad():
            y = torch_f.conv2d(x_t, conv1_w_t) + conv1_b_t
            y = torch.clamp_min(y, 0.0)
            y = torch_f.avg_pool2d(y, kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
            y = torch_f.conv2d(y, conv2_w_t) + conv2_b_t
            y = torch.clamp_min(y, 0.0)
            y = torch_f.avg_pool2d(y, kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
            y = y.permute(0, 2, 3, 1).contiguous().view(x.shape[0], FC1_IN_FEATURES)
            y = torch.clamp_min(y.matmul(fc1_w_t) + fc1_b_t, 0.0)
            y = torch.clamp_min(y.matmul(fc2_w_t) + fc2_b_t, 0.0)
            return y.matmul(fc3_w_t) + fc3_b_t

    return run


def make_torch_runner_int32(
    x: npt.NDArray[np.int32],
    weights: LeNet5Weights,
) -> Callable[[], Any]:
    if torch is None or torch_f is None:
        raise RuntimeError("torch is not available")

    x_t = torch.from_numpy(np.ascontiguousarray(x))
    conv1_w_t = torch.from_numpy(np.ascontiguousarray(weights.conv1_w))
    conv1_b_t = torch.from_numpy(np.ascontiguousarray(weights.conv1_b)).view(1, -1, 1, 1)
    conv2_w_t = torch.from_numpy(np.ascontiguousarray(weights.conv2_w))
    conv2_b_t = torch.from_numpy(np.ascontiguousarray(weights.conv2_b)).view(1, -1, 1, 1)
    fc1_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc1_w))
    fc1_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc1_b))
    fc2_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc2_w))
    fc2_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc2_b))
    fc3_w_t = torch.from_numpy(np.ascontiguousarray(weights.fc3_w))
    fc3_b_t = torch.from_numpy(np.ascontiguousarray(weights.fc3_b))

    def run() -> Any:
        with torch.no_grad():
            y = torch_f.conv2d(x_t, conv1_w_t) + conv1_b_t
            y = torch.clamp_min(y, 0)
            y = torch_avgpool2d_int32(y)
            y = torch_f.conv2d(y, conv2_w_t) + conv2_b_t
            y = torch.clamp_min(y, 0)
            y = torch_avgpool2d_int32(y)
            y = y.permute(0, 2, 3, 1).contiguous().view(x.shape[0], FC1_IN_FEATURES)
            y = torch.clamp_min(y.matmul(fc1_w_t) + fc1_b_t, 0)
            y = torch.clamp_min(y.matmul(fc2_w_t) + fc2_b_t, 0)
            return y.matmul(fc3_w_t) + fc3_b_t

    return run


def make_lenet5_problem_fp32(
    *,
    batch: int,
    seed: int,
) -> tuple[npt.NDArray[np.float32], LeNet5Weights]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).astype(np.float32)
    weights = LeNet5Weights(
        conv1_w=rng.uniform(-1.0, 1.0, size=(CONV1_OUT_CHANNELS, INPUT_CHANNELS, KERNEL, KERNEL)).astype(np.float32),
        conv1_b=rng.uniform(-0.5, 0.5, size=(CONV1_OUT_CHANNELS,)).astype(np.float32),
        conv2_w=rng.uniform(
            -1.0,
            1.0,
            size=(CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS, KERNEL, KERNEL),
        ).astype(np.float32),
        conv2_b=rng.uniform(-0.5, 0.5, size=(CONV2_OUT_CHANNELS,)).astype(np.float32),
        fc1_w=rng.uniform(-1.0, 1.0, size=(FC1_IN_FEATURES, FC1_OUT_FEATURES)).astype(np.float32),
        fc1_b=rng.uniform(-0.5, 0.5, size=(FC1_OUT_FEATURES,)).astype(np.float32),
        fc2_w=rng.uniform(-1.0, 1.0, size=(FC1_OUT_FEATURES, FC2_OUT_FEATURES)).astype(np.float32),
        fc2_b=rng.uniform(-0.5, 0.5, size=(FC2_OUT_FEATURES,)).astype(np.float32),
        fc3_w=rng.uniform(-1.0, 1.0, size=(FC2_OUT_FEATURES, FC3_OUT_FEATURES)).astype(np.float32),
        fc3_b=rng.uniform(-0.5, 0.5, size=(FC3_OUT_FEATURES,)).astype(np.float32),
    )
    return x, weights


def _int32_stage_inputs_within_contract(
    x: npt.NDArray[np.int32],
    weights: LeNet5Weights,
    reference: LeNet5Reference,
) -> bool:
    if max(
        _max_abs_int(x),
        _max_abs_int(weights.conv1_w.astype(np.int32)),
        _max_abs_int(weights.conv2_w.astype(np.int32)),
        _max_abs_int(weights.fc1_w.astype(np.int32)),
        _max_abs_int(weights.fc2_w.astype(np.int32)),
        _max_abs_int(weights.fc3_w.astype(np.int32)),
    ) >= SIGNED_24BIT_LIMIT:
        return False

    stage_inputs = [reference.pool1, reference.pool2, reference.fc1, reference.fc2]
    return all(_max_abs_int(stage.astype(np.int32)) < SIGNED_24BIT_LIMIT for stage in stage_inputs)


def make_lenet5_problem_int32(
    *,
    batch: int,
    seed: int,
) -> tuple[npt.NDArray[np.int32], LeNet5Weights, LeNet5Reference]:
    for attempt in range(128):
        rng = np.random.default_rng(seed + attempt)
        x = rng.integers(-2, 3, size=(batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.int32)
        weights = LeNet5Weights(
            conv1_w=rng.integers(-2, 3, size=(CONV1_OUT_CHANNELS, INPUT_CHANNELS, KERNEL, KERNEL), dtype=np.int32),
            conv1_b=rng.integers(-8, 9, size=(CONV1_OUT_CHANNELS,), dtype=np.int32),
            conv2_w=rng.integers(-2, 3, size=(CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS, KERNEL, KERNEL), dtype=np.int32),
            conv2_b=rng.integers(-16, 17, size=(CONV2_OUT_CHANNELS,), dtype=np.int32),
            fc1_w=rng.integers(-1, 2, size=(FC1_IN_FEATURES, FC1_OUT_FEATURES), dtype=np.int32),
            fc1_b=rng.integers(-32, 33, size=(FC1_OUT_FEATURES,), dtype=np.int32),
            fc2_w=rng.integers(-1, 2, size=(FC1_OUT_FEATURES, FC2_OUT_FEATURES), dtype=np.int32),
            fc2_b=rng.integers(-32, 33, size=(FC2_OUT_FEATURES,), dtype=np.int32),
            fc3_w=rng.integers(-1, 2, size=(FC2_OUT_FEATURES, FC3_OUT_FEATURES), dtype=np.int32),
            fc3_b=rng.integers(-32, 33, size=(FC3_OUT_FEATURES,), dtype=np.int32),
        )
        reference = reference_lenet5_int32(x, weights)
        if _int32_stage_inputs_within_contract(x, weights, reference):
            return x, weights, reference
    raise RuntimeError("failed to generate an int32 LeNet-5 benchmark problem that satisfies the smul24 contract")


def int32_stage_bounds(
    x: npt.NDArray[np.int32],
    reference: LeNet5Reference,
) -> dict[str, int]:
    return {
        "input": _max_abs_int(x),
        "pool1": _max_abs_int(reference.pool1.astype(np.int32)),
        "pool2_flat": _max_abs_int(reference.flat.astype(np.int32)),
        "fc1": _max_abs_int(reference.fc1.astype(np.int32)),
        "fc2": _max_abs_int(reference.fc2.astype(np.int32)),
    }


def _pad_matrix(
    matrix: npt.NDArray[np.generic],
    *,
    rows: int,
    cols: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.generic]:
    padded = np.zeros((rows, cols), dtype=dtype)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return np.ascontiguousarray(padded)


def _pad_bias(
    bias: npt.NDArray[np.generic],
    *,
    size: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.generic]:
    padded = np.zeros(size, dtype=dtype)
    padded[: bias.shape[0]] = bias
    return np.ascontiguousarray(padded)


def _array_matrix_view(matrix: Array[np.generic]) -> MatrixView:
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {matrix.shape}")
    return MatrixView(
        rows=int(matrix.shape[0]),
        cols=int(matrix.shape[1]),
        row_stride=int(matrix.strides[0]),
        base_addr=int(matrix.addresses().item(0)),
        itemsize=int(matrix.dtype.itemsize),
    )


def _build_tiled_matmul_dispatch(
    drv: Driver,
    code: Array[np.uint64],
    a_view: MatrixView,
    b_view: MatrixView,
    c_view: MatrixView,
    *,
    q: int,
) -> Dispatch:
    if a_view.cols != b_view.rows:
        raise ValueError("matrix shapes do not align")
    if c_view.rows != a_view.rows or c_view.cols != b_view.cols:
        raise ValueError("output matrix shape does not match the input matrices")
    if a_view.rows % P_TILE != 0 or q % Q_TILE != 0 or b_view.cols % R_TILE != 0:
        raise ValueError("tiled matmul buffers must already be padded to the kernel tile sizes")

    uniforms = drv.alloc(7, dtype=np.uint32)
    uniforms[0] = a_view.row_stride
    uniforms[1] = a_view.base_addr
    uniforms[2] = b_view.row_stride
    uniforms[3] = b_view.base_addr
    uniforms[4] = c_view.row_stride
    uniforms[5] = c_view.base_addr
    uniforms[6] = q
    return Dispatch(
        code=code,
        uniforms=uniforms,
        thread=(a_view.rows // P_TILE) * (b_view.cols // R_TILE),
        workgroup=(b_view.cols // R_TILE, a_view.rows // P_TILE, 1),
        wgs_per_sg=24,
    )


def _build_bias_dispatch(
    drv: Driver,
    code: Array[np.uint64],
    matrix: Array[np.generic],
    bias: Array[np.generic],
) -> Dispatch:
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[1] % 16 != 0 or matrix.shape[0] % 16 != 0:
        raise ValueError("bias-activation matrix shape must be padded to 16x16 tiles")
    uniforms = drv.alloc(3, dtype=np.uint32)
    uniforms[0] = matrix.strides[0]
    uniforms[1] = matrix.addresses().item(0)
    uniforms[2] = bias.addresses().item(0)
    return Dispatch(
        code=code,
        uniforms=uniforms,
        thread=(matrix.shape[0] // 16) * (matrix.shape[1] // 16),
        workgroup=(matrix.shape[1] // 16, matrix.shape[0] // 16, 1),
        wgs_per_sg=24,
    )


def _build_stream_dispatch(
    drv: Driver,
    code: Array[np.uint64],
    *,
    num_qpus: int,
    iterations: int,
    meta: Array[np.uint32],
    dst_addr: int,
    extra_uniforms: list[int] | None = None,
) -> Dispatch:
    extra_uniforms = [] if extra_uniforms is None else extra_uniforms
    uniforms = drv.alloc(3 + len(extra_uniforms), dtype=np.uint32)
    uniforms[0] = iterations
    uniforms[1] = meta.addresses()[0]
    uniforms[2] = dst_addr
    for idx, value in enumerate(extra_uniforms, start=3):
        uniforms[idx] = value
    return Dispatch(code=code, uniforms=uniforms, thread=num_qpus)


def _execute_dispatch(drv: Driver, dispatch: Dispatch) -> None:
    kwargs: dict[str, Any] = {
        "code": dispatch.code,
        "local_invocation": (16, 1, 1),
        "uniforms": dispatch.uniforms.addresses().item(0),
        "wgs_per_sg": dispatch.wgs_per_sg,
        "thread": dispatch.thread,
    }
    if dispatch.workgroup is not None:
        kwargs["workgroup"] = dispatch.workgroup
    drv.execute(**kwargs)


def build_nchw_conv_lowering_meta(
    *,
    base_addr: int,
    batch_stride: int,
    channel_stride: int,
    row_stride: int,
    col_stride: int,
    batch: int,
    in_channels: int,
    in_height: int,
    in_width: int,
    kernel_height: int,
    kernel_width: int,
    q_padded: int,
    zero_addr: int,
) -> npt.NDArray[np.uint32]:
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1
    meta = np.empty((batch * out_height * out_width, q_padded), dtype=np.uint32)
    row_idx = 0
    for n in range(batch):
        batch_base = base_addr + n * batch_stride
        for oy in range(out_height):
            for ox in range(out_width):
                col_idx = 0
                for ic in range(in_channels):
                    channel_base = batch_base + ic * channel_stride
                    for ky in range(kernel_height):
                        for kx in range(kernel_width):
                            meta[row_idx, col_idx] = channel_base + (oy + ky) * row_stride + (ox + kx) * col_stride
                            col_idx += 1
                meta[row_idx, col_idx:q_padded] = zero_addr
                row_idx += 1
    return np.ascontiguousarray(meta.reshape(-1))


def build_matrix_conv_lowering_meta(
    *,
    base_addr: int,
    row_stride: int,
    itemsize: int,
    batch: int,
    in_channels: int,
    in_height: int,
    in_width: int,
    kernel_height: int,
    kernel_width: int,
    q_padded: int,
    zero_addr: int,
) -> npt.NDArray[np.uint32]:
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1
    meta = np.empty((batch * out_height * out_width, q_padded), dtype=np.uint32)
    rows_per_batch = in_height * in_width
    row_idx = 0
    for n in range(batch):
        batch_row_base = n * rows_per_batch
        for oy in range(out_height):
            for ox in range(out_width):
                col_idx = 0
                for ic in range(in_channels):
                    for ky in range(kernel_height):
                        src_row_idx = batch_row_base + (oy + ky) * in_width + ox
                        src_base = base_addr + src_row_idx * row_stride + ic * itemsize
                        for kx in range(kernel_width):
                            meta[row_idx, col_idx] = src_base + kx * row_stride
                            col_idx += 1
                meta[row_idx, col_idx:q_padded] = zero_addr
                row_idx += 1
    return np.ascontiguousarray(meta.reshape(-1))


def build_matrix_pool_meta(
    *,
    base_addr: int,
    row_stride: int,
    itemsize: int,
    batch: int,
    in_height: int,
    in_width: int,
    channels_actual: int,
    channels_padded: int,
    zero_addr: int,
) -> tuple[npt.NDArray[np.uint32], int, int]:
    if in_height % 2 != 0 or in_width % 2 != 0:
        raise ValueError("avgpool expects even H/W")
    out_height = in_height // 2
    out_width = in_width // 2
    meta = np.empty((batch * out_height * out_width, channels_padded), dtype=np.uint32)
    rows_per_batch = in_height * in_width
    row_idx = 0
    for n in range(batch):
        batch_row_base = n * rows_per_batch
        for oy in range(out_height):
            for ox in range(out_width):
                src_row_idx = batch_row_base + (2 * oy) * in_width + (2 * ox)
                src_base = base_addr + src_row_idx * row_stride
                for channel in range(channels_actual):
                    meta[row_idx, channel] = src_base + channel * itemsize
                meta[row_idx, channels_actual:channels_padded] = zero_addr
                row_idx += 1
    return np.ascontiguousarray(meta.reshape(-1)), row_stride, in_width * row_stride


@qpu
def qpu_gather_copy_words(asm: Assembly, *, num_qpus: int) -> None:
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_qpu_num = rf3
    reg_offset = rf4
    reg_word_stride = rf5
    reg_src = rf10
    reg_val = rf11

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_word_stride, 1)
        shl(reg_word_stride, reg_word_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_word_stride, 3)
        shl(reg_word_stride, reg_word_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_meta, reg_meta, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    with loop as l:  # noqa: E741
        mov(tmua, reg_meta, sig=thrsw).add(reg_meta, reg_meta, reg_word_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_src))

        mov(tmua, reg_src, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_val))

        mov(tmud, reg_val)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_word_stride)
        tmuwt()

        l.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def emit_trunc_div4_int(dst: Register, src: Register, sign: Register, bias: Register, shift31: Register) -> None:
    asr(sign, src, shift31)
    band(bias, sign, 3)
    add(dst, src, bias)
    asr(dst, dst, 2)


@qpu
def qpu_pool2d_matrix_fp32(asm: Assembly, *, mode: str, num_qpus: int) -> None:
    if mode not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_x_stride = rf3
    reg_y_stride = rf4
    reg_qpu_num = rf5
    reg_offset = rf6
    reg_word_stride = rf7
    reg_base = rf10
    reg_v0 = rf11
    reg_v1 = rf12
    reg_v2 = rf13
    reg_v3 = rf14
    reg_tmp = rf15
    reg_out = rf16

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_x_stride))
    nop(sig=ldunifrf(reg_y_stride))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_word_stride, 1)
        shl(reg_word_stride, reg_word_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_word_stride, 3)
        shl(reg_word_stride, reg_word_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_meta, reg_meta, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    with loop as l:  # noqa: E741
        mov(tmua, reg_meta, sig=thrsw).add(reg_meta, reg_meta, reg_word_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_base))

        mov(tmua, reg_base, sig=thrsw)
        add(rf31, reg_base, reg_x_stride)
        mov(tmua, rf31, sig=thrsw)
        add(reg_tmp, reg_base, reg_y_stride)
        mov(tmua, reg_tmp, sig=thrsw)
        add(rf31, reg_tmp, reg_x_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        nop(sig=ldtmu(reg_v1))
        nop(sig=ldtmu(reg_v2))
        nop()
        nop(sig=ldtmu(reg_v3))

        if mode == "max":
            fmax(reg_out, reg_v0, reg_v1)
            fmax(reg_out, reg_out, reg_v2)
            fmax(reg_out, reg_out, reg_v3)
        else:
            fadd(reg_out, reg_v0, reg_v1)
            fadd(reg_tmp, reg_v2, reg_v3)
            fadd(reg_out, reg_out, reg_tmp)
            fmul(reg_out, reg_out, 0.25)

        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_word_stride)
        tmuwt()

        l.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_pool2d_matrix_int32(asm: Assembly, *, mode: str, num_qpus: int) -> None:
    if mode not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_x_stride = rf3
    reg_y_stride = rf4
    reg_qpu_num = rf5
    reg_offset = rf6
    reg_word_stride = rf7
    reg_shift31 = rf8
    reg_base = rf10
    reg_v0 = rf11
    reg_v1 = rf12
    reg_v2 = rf13
    reg_v3 = rf14
    reg_tmp = rf15
    reg_out = rf16
    reg_sign = rf17
    reg_bias = rf18

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_x_stride))
    nop(sig=ldunifrf(reg_y_stride))
    mov(reg_shift31, -1)

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_word_stride, 1)
        shl(reg_word_stride, reg_word_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_word_stride, 3)
        shl(reg_word_stride, reg_word_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_meta, reg_meta, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    with loop as l:  # noqa: E741
        mov(tmua, reg_meta, sig=thrsw).add(reg_meta, reg_meta, reg_word_stride)
        nop()
        nop()
        nop(sig=ldtmu(reg_base))

        mov(tmua, reg_base, sig=thrsw)
        add(rf31, reg_base, reg_x_stride)
        mov(tmua, rf31, sig=thrsw)
        add(reg_tmp, reg_base, reg_y_stride)
        mov(tmua, reg_tmp, sig=thrsw)
        add(rf31, reg_tmp, reg_x_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        nop(sig=ldtmu(reg_v1))
        nop(sig=ldtmu(reg_v2))
        nop()
        nop(sig=ldtmu(reg_v3))

        if mode == "max":
            imax(reg_out, reg_v0, reg_v1)
            imax(reg_out, reg_out, reg_v2)
            imax(reg_out, reg_out, reg_v3)
        else:
            add(reg_out, reg_v0, reg_v1)
            add(reg_tmp, reg_v2, reg_v3)
            add(reg_out, reg_out, reg_tmp)
            emit_trunc_div4_int(reg_out, reg_out, reg_sign, reg_bias, reg_shift31)

        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_word_stride)
        tmuwt()

        l.b(cond="na0")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


DTYPE_CONFIGS: dict[str, DTypeConfig] = {
    "fp32": DTypeConfig(
        name="fp32",
        np_dtype=np.dtype(np.float32),
        matmul_kernel=qpu_tiledconv2d_fp32,
        bias_kernel=qpu_tiled_bias_activation_fp32,
        pool_kernel=qpu_pool2d_matrix_fp32,
    ),
    "int32": DTypeConfig(
        name="int32",
        np_dtype=np.dtype(np.int32),
        matmul_kernel=qpu_tiledconv2d_int32,
        bias_kernel=qpu_tiled_bias_activation_int32,
        pool_kernel=qpu_pool2d_matrix_int32,
    ),
}


class LeNet5QpuExecutor:
    batch: int
    num_qpus: int
    config: DTypeConfig
    _drv: Driver

    def __init__(
        self,
        config: DTypeConfig,
        weights: LeNet5Weights,
        *,
        batch: int,
        num_qpus: int,
    ) -> None:
        if num_qpus not in (1, 12):
            raise ValueError("num_qpus must be 1 or 12")

        batch_tile = lcm(16, num_qpus)
        if batch % batch_tile != 0:
            raise ValueError(
                f"batch must be a multiple of {batch_tile} for the current "
                "multi-QPU lowering/pool schedule"
            )

        dtype = config.np_dtype
        if any(
            arr.dtype != dtype
            for arr in (
                weights.conv1_w,
                weights.conv1_b,
                weights.conv2_w,
                weights.conv2_b,
                weights.fc1_w,
                weights.fc1_b,
                weights.fc2_w,
                weights.fc2_b,
                weights.fc3_w,
                weights.fc3_b,
            )
        ):
            raise ValueError(f"{config.name} executor expects all weights and biases to use dtype {dtype}")

        self.batch = int(batch)
        self.num_qpus = int(num_qpus)
        self.config = config

        self.conv1_p = self.batch * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH
        self.conv1_q = INPUT_CHANNELS * KERNEL * KERNEL
        self.conv1_q_padded = _round_up(self.conv1_q, Q_TILE)
        self.conv1_r_padded = _round_up(CONV1_OUT_CHANNELS, R_TILE)

        self.pool1_p = self.batch * POOL1_OUT_HEIGHT * POOL1_OUT_WIDTH
        self.pool1_channels_padded = self.conv1_r_padded

        self.conv2_p = self.batch * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH
        self.conv2_q = CONV1_OUT_CHANNELS * KERNEL * KERNEL
        self.conv2_q_padded = _round_up(self.conv2_q, Q_TILE)
        self.conv2_r_padded = _round_up(CONV2_OUT_CHANNELS, R_TILE)

        self.pool2_p = self.batch * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH
        self.pool2_channels_padded = self.conv2_r_padded

        self.fc1_r_padded = _round_up(FC1_OUT_FEATURES, R_TILE)
        self.fc2_q_padded = self.fc1_r_padded
        self.fc2_r_padded = _round_up(FC2_OUT_FEATURES, R_TILE)
        self.fc3_q_padded = self.fc2_r_padded
        self.fc3_r_padded = _round_up(FC3_OUT_FEATURES, R_TILE)

        itemsize = dtype.itemsize
        data_area_size = (
            batch * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH
            + 1
            + self.conv1_p * self.conv1_q_padded
            + self.conv1_q_padded * self.conv1_r_padded
            + self.conv1_p * self.conv1_r_padded
            + self.conv1_r_padded
            + self.pool1_p * self.pool1_channels_padded
            + self.conv2_p * self.conv2_q_padded
            + self.conv2_q_padded * self.conv2_r_padded
            + self.conv2_p * self.conv2_r_padded
            + self.conv2_r_padded
            + self.pool2_p * self.pool2_channels_padded
            + FC1_IN_FEATURES * self.fc1_r_padded
            + self.batch * self.fc1_r_padded
            + self.fc1_r_padded
            + self.fc2_q_padded * self.fc2_r_padded
            + self.batch * self.fc2_r_padded
            + self.fc2_r_padded
            + self.fc3_q_padded * self.fc3_r_padded
            + self.batch * self.fc3_r_padded
            + self.fc3_r_padded
        ) * itemsize + (
            self.conv1_p * self.conv1_q_padded
            + self.pool1_p * self.pool1_channels_padded
            + self.conv2_p * self.conv2_q_padded
            + self.pool2_p * self.pool2_channels_padded
        ) * 4 + (1 << 20)

        self._drv = Driver(data_area_size=data_area_size)
        self._gather_code = self._drv.program(qpu_gather_copy_words, num_qpus=self.num_qpus)
        self._pool_code = self._drv.program(config.pool_kernel, mode="avg", num_qpus=self.num_qpus)
        self._matmul_code = self._drv.program(config.matmul_kernel)
        self._bias_relu_code = self._drv.program(config.bias_kernel, use_bias=True, apply_relu=True)
        self._bias_only_code = self._drv.program(config.bias_kernel, use_bias=True, apply_relu=False)

        self._input_dev = self._drv.alloc((self.batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH), dtype=dtype)
        self._zero_dev = self._drv.alloc(1, dtype=dtype)
        self._zero_dev[:] = 0

        self._conv1_col_dev = self._drv.alloc((self.conv1_p, self.conv1_q_padded), dtype=dtype)
        self._conv1_w_dev = self._drv.alloc((self.conv1_q_padded, self.conv1_r_padded), dtype=dtype)
        self._conv1_out_dev = self._drv.alloc((self.conv1_p, self.conv1_r_padded), dtype=dtype)
        self._conv1_bias_dev = self._drv.alloc(self.conv1_r_padded, dtype=dtype)

        self._pool1_out_dev = self._drv.alloc((self.pool1_p, self.pool1_channels_padded), dtype=dtype)

        self._conv2_col_dev = self._drv.alloc((self.conv2_p, self.conv2_q_padded), dtype=dtype)
        self._conv2_w_dev = self._drv.alloc((self.conv2_q_padded, self.conv2_r_padded), dtype=dtype)
        self._conv2_out_dev = self._drv.alloc((self.conv2_p, self.conv2_r_padded), dtype=dtype)
        self._conv2_bias_dev = self._drv.alloc(self.conv2_r_padded, dtype=dtype)

        self._pool2_out_dev = self._drv.alloc((self.pool2_p, self.pool2_channels_padded), dtype=dtype)

        self._fc1_w_dev = self._drv.alloc((FC1_IN_FEATURES, self.fc1_r_padded), dtype=dtype)
        self._fc1_out_dev = self._drv.alloc((self.batch, self.fc1_r_padded), dtype=dtype)
        self._fc1_bias_dev = self._drv.alloc(self.fc1_r_padded, dtype=dtype)

        self._fc2_w_dev = self._drv.alloc((self.fc2_q_padded, self.fc2_r_padded), dtype=dtype)
        self._fc2_out_dev = self._drv.alloc((self.batch, self.fc2_r_padded), dtype=dtype)
        self._fc2_bias_dev = self._drv.alloc(self.fc2_r_padded, dtype=dtype)

        self._fc3_w_dev = self._drv.alloc((self.fc3_q_padded, self.fc3_r_padded), dtype=dtype)
        self._output_dev = self._drv.alloc((self.batch, self.fc3_r_padded), dtype=dtype)
        self._fc3_bias_dev = self._drv.alloc(self.fc3_r_padded, dtype=dtype)

        self._conv1_out_dev[:] = 0
        self._pool1_out_dev[:] = 0
        self._conv2_out_dev[:] = 0
        self._pool2_out_dev[:] = 0
        self._fc1_out_dev[:] = 0
        self._fc2_out_dev[:] = 0
        self._output_dev[:] = 0

        self._conv1_w_dev[:] = _pad_matrix(
            reshape_weight_oihw_to_gemm(weights.conv1_w),
            rows=self.conv1_q_padded,
            cols=self.conv1_r_padded,
            dtype=dtype,
        )
        self._conv1_bias_dev[:] = _pad_bias(weights.conv1_b, size=self.conv1_r_padded, dtype=dtype)

        self._conv2_w_dev[:] = _pad_matrix(
            reshape_weight_oihw_to_gemm(weights.conv2_w),
            rows=self.conv2_q_padded,
            cols=self.conv2_r_padded,
            dtype=dtype,
        )
        self._conv2_bias_dev[:] = _pad_bias(weights.conv2_b, size=self.conv2_r_padded, dtype=dtype)

        self._fc1_w_dev[:] = _pad_matrix(weights.fc1_w, rows=FC1_IN_FEATURES, cols=self.fc1_r_padded, dtype=dtype)
        self._fc1_bias_dev[:] = _pad_bias(weights.fc1_b, size=self.fc1_r_padded, dtype=dtype)
        self._fc2_w_dev[:] = _pad_matrix(weights.fc2_w, rows=self.fc2_q_padded, cols=self.fc2_r_padded, dtype=dtype)
        self._fc2_bias_dev[:] = _pad_bias(weights.fc2_b, size=self.fc2_r_padded, dtype=dtype)
        self._fc3_w_dev[:] = _pad_matrix(weights.fc3_w, rows=self.fc3_q_padded, cols=self.fc3_r_padded, dtype=dtype)
        self._fc3_bias_dev[:] = _pad_bias(weights.fc3_b, size=self.fc3_r_padded, dtype=dtype)

        zero_addr = int(self._zero_dev.addresses()[0])
        conv1_meta_host = build_nchw_conv_lowering_meta(
            base_addr=int(self._input_dev.addresses().item(0)),
            batch_stride=int(self._input_dev.strides[0]),
            channel_stride=int(self._input_dev.strides[1]),
            row_stride=int(self._input_dev.strides[2]),
            col_stride=int(self._input_dev.strides[3]),
            batch=self.batch,
            in_channels=INPUT_CHANNELS,
            in_height=INPUT_HEIGHT,
            in_width=INPUT_WIDTH,
            kernel_height=KERNEL,
            kernel_width=KERNEL,
            q_padded=self.conv1_q_padded,
            zero_addr=zero_addr,
        )
        pool1_meta_host, pool1_x_stride, pool1_y_stride = build_matrix_pool_meta(
            base_addr=int(self._conv1_out_dev.addresses().item(0)),
            row_stride=int(self._conv1_out_dev.strides[0]),
            itemsize=itemsize,
            batch=self.batch,
            in_height=CONV1_OUT_HEIGHT,
            in_width=CONV1_OUT_WIDTH,
            channels_actual=CONV1_OUT_CHANNELS,
            channels_padded=self.pool1_channels_padded,
            zero_addr=zero_addr,
        )
        conv2_meta_host = build_matrix_conv_lowering_meta(
            base_addr=int(self._pool1_out_dev.addresses().item(0)),
            row_stride=int(self._pool1_out_dev.strides[0]),
            itemsize=itemsize,
            batch=self.batch,
            in_channels=CONV1_OUT_CHANNELS,
            in_height=POOL1_OUT_HEIGHT,
            in_width=POOL1_OUT_WIDTH,
            kernel_height=KERNEL,
            kernel_width=KERNEL,
            q_padded=self.conv2_q_padded,
            zero_addr=zero_addr,
        )
        pool2_meta_host, pool2_x_stride, pool2_y_stride = build_matrix_pool_meta(
            base_addr=int(self._conv2_out_dev.addresses().item(0)),
            row_stride=int(self._conv2_out_dev.strides[0]),
            itemsize=itemsize,
            batch=self.batch,
            in_height=CONV2_OUT_HEIGHT,
            in_width=CONV2_OUT_WIDTH,
            channels_actual=CONV2_OUT_CHANNELS,
            channels_padded=self.pool2_channels_padded,
            zero_addr=zero_addr,
        )

        self._conv1_meta_dev = self._drv.alloc(conv1_meta_host.shape, dtype=np.uint32)
        self._pool1_meta_dev = self._drv.alloc(pool1_meta_host.shape, dtype=np.uint32)
        self._conv2_meta_dev = self._drv.alloc(conv2_meta_host.shape, dtype=np.uint32)
        self._pool2_meta_dev = self._drv.alloc(pool2_meta_host.shape, dtype=np.uint32)
        self._conv1_meta_dev[:] = conv1_meta_host
        self._pool1_meta_dev[:] = pool1_meta_host
        self._conv2_meta_dev[:] = conv2_meta_host
        self._pool2_meta_dev[:] = pool2_meta_host

        self._validate_stream_geometry()

        self._conv1_gather_dispatch = _build_stream_dispatch(
            self._drv,
            self._gather_code,
            num_qpus=self.num_qpus,
            iterations=conv1_meta_host.size // (16 * self.num_qpus),
            meta=self._conv1_meta_dev,
            dst_addr=int(self._conv1_col_dev.addresses().item(0)),
        )
        self._pool1_dispatch = _build_stream_dispatch(
            self._drv,
            self._pool_code,
            num_qpus=self.num_qpus,
            iterations=pool1_meta_host.size // (16 * self.num_qpus),
            meta=self._pool1_meta_dev,
            dst_addr=int(self._pool1_out_dev.addresses().item(0)),
            extra_uniforms=[pool1_x_stride, pool1_y_stride],
        )
        self._conv2_gather_dispatch = _build_stream_dispatch(
            self._drv,
            self._gather_code,
            num_qpus=self.num_qpus,
            iterations=conv2_meta_host.size // (16 * self.num_qpus),
            meta=self._conv2_meta_dev,
            dst_addr=int(self._conv2_col_dev.addresses().item(0)),
        )
        self._pool2_dispatch = _build_stream_dispatch(
            self._drv,
            self._pool_code,
            num_qpus=self.num_qpus,
            iterations=pool2_meta_host.size // (16 * self.num_qpus),
            meta=self._pool2_meta_dev,
            dst_addr=int(self._pool2_out_dev.addresses().item(0)),
            extra_uniforms=[pool2_x_stride, pool2_y_stride],
        )

        self._conv1_linear_dispatch = _build_tiled_matmul_dispatch(
            self._drv,
            self._matmul_code,
            _array_matrix_view(self._conv1_col_dev),
            _array_matrix_view(self._conv1_w_dev),
            _array_matrix_view(self._conv1_out_dev),
            q=self.conv1_q_padded,
        )
        self._conv1_bias_dispatch = _build_bias_dispatch(
            self._drv,
            self._bias_relu_code,
            self._conv1_out_dev,
            self._conv1_bias_dev,
        )

        self._conv2_linear_dispatch = _build_tiled_matmul_dispatch(
            self._drv,
            self._matmul_code,
            _array_matrix_view(self._conv2_col_dev),
            _array_matrix_view(self._conv2_w_dev),
            _array_matrix_view(self._conv2_out_dev),
            q=self.conv2_q_padded,
        )
        self._conv2_bias_dispatch = _build_bias_dispatch(
            self._drv,
            self._bias_relu_code,
            self._conv2_out_dev,
            self._conv2_bias_dev,
        )

        fc1_input_view = MatrixView(
            rows=self.batch,
            cols=FC1_IN_FEATURES,
            row_stride=FC1_IN_FEATURES * itemsize,
            base_addr=int(self._pool2_out_dev.addresses().item(0)),
            itemsize=itemsize,
        )
        self._fc1_linear_dispatch = _build_tiled_matmul_dispatch(
            self._drv,
            self._matmul_code,
            fc1_input_view,
            _array_matrix_view(self._fc1_w_dev),
            _array_matrix_view(self._fc1_out_dev),
            q=FC1_IN_FEATURES,
        )
        self._fc1_bias_dispatch = _build_bias_dispatch(
            self._drv,
            self._bias_relu_code,
            self._fc1_out_dev,
            self._fc1_bias_dev,
        )

        self._fc2_linear_dispatch = _build_tiled_matmul_dispatch(
            self._drv,
            self._matmul_code,
            _array_matrix_view(self._fc1_out_dev),
            _array_matrix_view(self._fc2_w_dev),
            _array_matrix_view(self._fc2_out_dev),
            q=self.fc2_q_padded,
        )
        self._fc2_bias_dispatch = _build_bias_dispatch(
            self._drv,
            self._bias_relu_code,
            self._fc2_out_dev,
            self._fc2_bias_dev,
        )

        self._fc3_linear_dispatch = _build_tiled_matmul_dispatch(
            self._drv,
            self._matmul_code,
            _array_matrix_view(self._fc2_out_dev),
            _array_matrix_view(self._fc3_w_dev),
            _array_matrix_view(self._output_dev),
            q=self.fc3_q_padded,
        )
        self._fc3_bias_dispatch = _build_bias_dispatch(
            self._drv,
            self._bias_only_code,
            self._output_dev,
            self._fc3_bias_dev,
        )

    def _validate_stream_geometry(self) -> None:
        required_multiple = 16 * self.num_qpus
        stream_sizes = {
            "conv1 lowering": self.conv1_p * self.conv1_q_padded,
            "pool1": self.pool1_p * self.pool1_channels_padded,
            "conv2 lowering": self.conv2_p * self.conv2_q_padded,
            "pool2": self.pool2_p * self.pool2_channels_padded,
        }
        for name, size in stream_sizes.items():
            if size % required_multiple != 0:
                raise ValueError(f"{name} element count must be a multiple of {required_multiple}")

    def close(self) -> None:
        self._drv.close()

    def __enter__(self) -> LeNet5QpuExecutor:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def upload_input(self, x: npt.NDArray[np.generic]) -> None:
        if x.shape != self._input_dev.shape:
            raise ValueError(f"expected input shape {self._input_dev.shape}, got {x.shape}")
        if x.dtype != self.config.np_dtype:
            raise ValueError(f"expected dtype {self.config.np_dtype}, got {x.dtype}")
        self._input_dev[:] = x

    def read_output(self) -> npt.NDArray[np.generic]:
        out = np.array(self._output_dev, copy=True)
        return np.ascontiguousarray(out[:, :FC3_OUT_FEATURES])

    def execute_conv1_gather(self) -> None:
        _execute_dispatch(self._drv, self._conv1_gather_dispatch)

    def execute_conv1_block(self) -> None:
        _execute_dispatch(self._drv, self._conv1_linear_dispatch)
        _execute_dispatch(self._drv, self._conv1_bias_dispatch)

    def execute_pool1(self) -> None:
        _execute_dispatch(self._drv, self._pool1_dispatch)

    def execute_conv2_gather(self) -> None:
        _execute_dispatch(self._drv, self._conv2_gather_dispatch)

    def execute_conv2_block(self) -> None:
        _execute_dispatch(self._drv, self._conv2_linear_dispatch)
        _execute_dispatch(self._drv, self._conv2_bias_dispatch)

    def execute_pool2(self) -> None:
        _execute_dispatch(self._drv, self._pool2_dispatch)

    def execute_fc1_block(self) -> None:
        _execute_dispatch(self._drv, self._fc1_linear_dispatch)
        _execute_dispatch(self._drv, self._fc1_bias_dispatch)

    def execute_fc2_block(self) -> None:
        _execute_dispatch(self._drv, self._fc2_linear_dispatch)
        _execute_dispatch(self._drv, self._fc2_bias_dispatch)

    def execute_fc3_block(self) -> None:
        _execute_dispatch(self._drv, self._fc3_linear_dispatch)
        _execute_dispatch(self._drv, self._fc3_bias_dispatch)

    def execute_pipeline(self) -> None:
        self.execute_conv1_gather()
        self.execute_conv1_block()
        self.execute_pool1()
        self.execute_conv2_gather()
        self.execute_conv2_block()
        self.execute_pool2()
        self.execute_fc1_block()
        self.execute_fc2_block()
        self.execute_fc3_block()

    def stage_groups(self) -> list[tuple[str, Callable[[], None]]]:
        return [
            ("conv1 gather", self.execute_conv1_gather),
            ("conv1 gemm+bias+relu", self.execute_conv1_block),
            ("pool1 avgpool", self.execute_pool1),
            ("conv2 gather", self.execute_conv2_gather),
            ("conv2 gemm+bias+relu", self.execute_conv2_block),
            ("pool2 avgpool", self.execute_pool2),
            ("fc1 gemm+bias+relu", self.execute_fc1_block),
            ("fc2 gemm+bias+relu", self.execute_fc2_block),
            ("fc3 gemm+bias", self.execute_fc3_block),
        ]


def _benchmark_qpu_pipeline(
    executor: LeNet5QpuExecutor,
    x: npt.NDArray[np.generic],
    expected: npt.NDArray[np.generic],
) -> tuple[npt.NDArray[np.generic], QpuBenchmarkStats, list[StageTiming]]:
    prep_sec = 0.0

    def run_total() -> npt.NDArray[np.generic]:
        executor.upload_input(x)
        executor.execute_pipeline()
        return executor.read_output()

    actual, cached_total_sec = _benchmark_callable(run_total, repeat=5)
    executor.upload_input(x)
    execute_only_sec = _benchmark_timing(executor.execute_pipeline, repeat=10)

    executor.upload_input(x)
    stage_timings: list[StageTiming] = []
    for label, fn in executor.stage_groups():
        stage_timings.append(StageTiming(label=label, sec=_benchmark_timing(fn, repeat=10)))

    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    stats = QpuBenchmarkStats(
        prep_sec=prep_sec,
        cached_total_sec=cached_total_sec,
        execute_only_sec=execute_only_sec,
        max_abs_error=float(np.max(diff)),
    )
    return actual, stats, stage_timings


def _print_qpu_stats(
    *,
    batch: int,
    numpy_sec: float,
    torch_sec: float | None,
    qpu_stats: QpuBenchmarkStats | None,
) -> None:
    print(f"numpy: {numpy_sec:.4f} sec, {lenet5_gops(batch, numpy_sec):.4f} Gop/s")
    if torch_sec is None:
        print("torch native lenet5: n/a")
    else:
        print(f"torch native lenet5: {torch_sec:.4f} sec, {lenet5_gops(batch, torch_sec):.4f} Gop/s")
    if qpu_stats is None:
        print("QPU: unavailable")
        return

    prep_cached_sec = qpu_stats.prep_sec + qpu_stats.cached_total_sec
    io_sec = max(qpu_stats.cached_total_sec - qpu_stats.execute_only_sec, 0.0)
    print(f"QPU host prep: {qpu_stats.prep_sec:.4f} sec")
    print(
        f"QPU cached total: {qpu_stats.cached_total_sec:.4f} sec, "
        f"{lenet5_gops(batch, qpu_stats.cached_total_sec):.4f} Gop/s"
    )
    print(
        f"QPU execute only: {qpu_stats.execute_only_sec:.4f} sec, "
        f"{lenet5_gops(batch, qpu_stats.execute_only_sec):.4f} Gop/s"
    )
    print(f"QPU prep+cached total: {prep_cached_sec:.4f} sec, {lenet5_gops(batch, prep_cached_sec):.4f} Gop/s")
    print(f"QPU upload+readback overhead inside cached total: {io_sec:.4f} sec")
    print(f"Maximum absolute error: {qpu_stats.max_abs_error}")


def _print_stage_breakdown(stage_timings: list[StageTiming]) -> None:
    print("QPU execute breakdown:")
    for timing in stage_timings:
        print(f"  {timing.label:<24} {timing.sec:.6f} sec")


def benchmark_lenet5_fp32(
    *,
    batch: int = 48,
    repeat: int = 3,
    seed: int = 0,
    num_qpus: int = 12,
) -> dict[str, float]:
    x, weights = make_lenet5_problem_fp32(batch=batch, seed=seed)
    reference = reference_lenet5_fp32(x, weights)
    expected = reference.output
    numpy_sec = _benchmark_timing(lambda: numpy_lenet5_fp32(x, weights), repeat=repeat)

    torch_sec = None
    if torch is not None and torch_f is not None:
        run_torch = make_torch_runner_fp32(x, weights)
        torch_output = run_torch().cpu().numpy()
        np.testing.assert_allclose(torch_output, expected, atol=5e-4, rtol=5e-4)
        torch_sec = _benchmark_timing(run_torch, repeat=repeat)

    print("==== LeNet-5 fp32 example ====")
    print(
        "Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> "
        "Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)"
    )
    print(f"Batch: {batch}, num_qpus: {num_qpus}")
    print("QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.")
    print("Steady-state CPU compute in the QPU path: none.")
    print(
        "Excluded one-time CPU setup: weight reshape/padding, metadata generation, "
        "kernel assembly, buffer allocation."
    )

    result: dict[str, float] = {"numpy_sec": numpy_sec}
    setup_start = getsec()
    try:
        executor = LeNet5QpuExecutor(DTYPE_CONFIGS["fp32"], weights, batch=batch, num_qpus=num_qpus)
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        _print_qpu_stats(batch=batch, numpy_sec=numpy_sec, torch_sec=torch_sec, qpu_stats=None)
        print()
        if torch_sec is not None:
            result["torch_sec"] = torch_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        _, qpu_stats, stage_timings = _benchmark_qpu_pipeline(executor, x, expected)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    _print_qpu_stats(batch=batch, numpy_sec=numpy_sec, torch_sec=torch_sec, qpu_stats=qpu_stats)
    _print_stage_breakdown(stage_timings)
    print()

    result.update(
        {
            "torch_sec": -1.0 if torch_sec is None else torch_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": qpu_stats.prep_sec,
            "qpu_cached_total_sec": qpu_stats.cached_total_sec,
            "qpu_execute_only_sec": qpu_stats.execute_only_sec,
            "max_abs_error": qpu_stats.max_abs_error,
        }
    )
    return result


def benchmark_lenet5_int32(
    *,
    batch: int = 48,
    repeat: int = 3,
    seed: int = 1,
    num_qpus: int = 12,
) -> dict[str, float]:
    x, weights, reference = make_lenet5_problem_int32(batch=batch, seed=seed)
    expected = reference.output
    numpy_sec = _benchmark_timing(lambda: numpy_lenet5_int32(x, weights), repeat=repeat)

    torch_sec = None
    if torch is not None and torch_f is not None:
        run_torch = make_torch_runner_int32(x, weights)
        torch_output = run_torch().cpu().numpy()
        np.testing.assert_array_equal(torch_output, expected)
        torch_sec = _benchmark_timing(run_torch, repeat=repeat)

    bounds = int32_stage_bounds(x, reference)
    print("==== LeNet-5 int32 example ====")
    print(
        "Architecture: Conv5x5(1->6) -> ReLU -> AvgPool -> Conv5x5(6->16) -> ReLU -> AvgPool -> "
        "Linear(400->120) -> ReLU -> Linear(120->84) -> ReLU -> Linear(84->10)"
    )
    print(f"Batch: {batch}, num_qpus: {num_qpus}")
    print("QPU steady-state layout: on-device gather lowering -> position-major matrices -> zero-copy flatten -> FC.")
    print("Steady-state CPU compute in the QPU path: none.")
    print(
        "Excluded one-time CPU setup: weight reshape/padding, metadata generation, "
        "kernel assembly, buffer allocation."
    )
    print(
        "Actual smul24 stage maxima: "
        f"input={bounds['input']}, pool1={bounds['pool1']}, pool2_flat={bounds['pool2_flat']}, "
        f"fc1={bounds['fc1']}, fc2={bounds['fc2']}"
    )

    result: dict[str, float] = {"numpy_sec": numpy_sec}
    setup_start = getsec()
    try:
        executor = LeNet5QpuExecutor(DTYPE_CONFIGS["int32"], weights, batch=batch, num_qpus=num_qpus)
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        _print_qpu_stats(batch=batch, numpy_sec=numpy_sec, torch_sec=torch_sec, qpu_stats=None)
        print()
        if torch_sec is not None:
            result["torch_sec"] = torch_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        _, qpu_stats, stage_timings = _benchmark_qpu_pipeline(executor, x, expected)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    _print_qpu_stats(batch=batch, numpy_sec=numpy_sec, torch_sec=torch_sec, qpu_stats=qpu_stats)
    _print_stage_breakdown(stage_timings)
    print()

    result.update(
        {
            "torch_sec": -1.0 if torch_sec is None else torch_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": qpu_stats.prep_sec,
            "qpu_cached_total_sec": qpu_stats.cached_total_sec,
            "qpu_execute_only_sec": qpu_stats.execute_only_sec,
            "max_abs_error": qpu_stats.max_abs_error,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a persistent LeNet-5-style QPU pipeline against NumPy and Torch"
    )
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-qpus", type=int, default=12, choices=(1, 12))
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPE_CONFIGS),
        default=["fp32", "int32"],
        help="Dtypes to benchmark",
    )
    args = parser.parse_args()

    for idx, dtype_name in enumerate(args.dtypes):
        if dtype_name == "fp32":
            benchmark_lenet5_fp32(batch=args.batch, repeat=args.repeat, seed=args.seed, num_qpus=args.num_qpus)
        else:
            benchmark_lenet5_int32(batch=args.batch, repeat=args.repeat, seed=args.seed + 1, num_qpus=args.num_qpus)
        if idx != len(args.dtypes) - 1:
            print()


if __name__ == "__main__":
    main()
