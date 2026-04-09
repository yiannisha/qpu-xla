from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass
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


def _load_sgemm_fast_module() -> Any:
    module_path = Path(__file__).with_name("sgemm_fast.py")
    spec = importlib.util.spec_from_file_location("_tiledmlp_sgemm_fast", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SGEMM_FAST = _load_sgemm_fast_module()
VC7_QPUS = int(_SGEMM_FAST.VC7_QPUS)
qpu_sgemm_rnn_reuse_a_x2 = _SGEMM_FAST.qpu_sgemm_rnn_reuse_a_x2
qpu_sgemm_rnn_reuse_a_x2_qpu_aware = _SGEMM_FAST.qpu_sgemm_rnn_reuse_a_x2_qpu_aware


def _load_tiledconv2d_module() -> Any:
    module_path = Path(__file__).with_name("tiledconv2d.py")
    spec = importlib.util.spec_from_file_location("_tiledmlp_tiledconv2d", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TILEDCONV2D = _load_tiledconv2d_module()
qpu_tiledconv2d_fp32 = _TILEDCONV2D.qpu_tiledconv2d_fp32
qpu_tiledconv2d_int32 = _TILEDCONV2D.qpu_tiledconv2d_int32

P_TILE = 16
Q_TILE = 4
R_TILE = 32
SIGNED_24BIT_LIMIT = 1 << 23
VALIDATED_LINEAR_R_TILE = 16


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def _round_up(value: int, tile: int) -> int:
    return ((value + tile - 1) // tile) * tile


def mlp_gops(
    batch: int,
    in_features: int,
    hidden_features: int,
    out_features: int,
    sec: float,
) -> float:
    ops = (
        2 * batch * in_features * hidden_features
        + batch * hidden_features
        + batch * hidden_features
        + 2 * batch * hidden_features * out_features
        + batch * out_features
    )
    return ops / sec * 1e-9


def linear_gops(
    batch: int,
    in_features: int,
    out_features: int,
    sec: float,
) -> float:
    ops = 2 * batch * in_features * out_features + batch * out_features
    return ops / sec * 1e-9


def relu_gops(
    batch: int,
    features: int,
    sec: float,
) -> float:
    ops = batch * features
    return ops / sec * 1e-9


def _benchmark_callable(
    fn,
    *,
    warmup: int = 1,
    repeat: int = 3,
):
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
    fn,
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


def _validate_int32_mlp_contract(
    x: npt.NDArray[np.int32],
    w1: npt.NDArray[np.int32],
    b1: npt.NDArray[np.int32],
    w2: npt.NDArray[np.int32],
    b2: npt.NDArray[np.int32],
) -> None:
    max_x = _max_abs_int(x)
    max_w1 = _max_abs_int(w1)
    max_b1 = _max_abs_int(b1)
    max_w2 = _max_abs_int(w2)
    max_b2 = _max_abs_int(b2)

    if max(max_x, max_w1, max_w2) >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 kernels use smul24; x, w1, and w2 must fit the signed 24-bit range")

    hidden_bound = x.shape[1] * max_x * max_w1 + max_b1
    if hidden_bound >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 MLP hidden activations may exceed the signed 24-bit range required by layer 2")

    output_bound = w2.shape[0] * hidden_bound * max_w2 + max_b2
    if output_bound > np.iinfo(np.int32).max:
        raise ValueError("int32 MLP outputs may exceed the signed int32 accumulation range")


def numpy_linear_fp32(
    x: npt.NDArray[np.float32],
    w: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(x.dot(w) + b, dtype=np.float32)


def numpy_relu_fp32(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(np.maximum(x, np.float32(0.0)), dtype=np.float32)


def numpy_mlp_naive(
    x: npt.NDArray[np.float32],
    w1: npt.NDArray[np.float32],
    b1: npt.NDArray[np.float32],
    w2: npt.NDArray[np.float32],
    b2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    hidden = numpy_linear_fp32(x, w1, b1)
    hidden = numpy_relu_fp32(hidden)
    return numpy_linear_fp32(hidden, w2, b2)


def reference_linear_int32(
    x: npt.NDArray[np.int32],
    w: npt.NDArray[np.int32],
    b: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    acc = x.astype(np.int64).dot(w.astype(np.int64))
    acc += b.astype(np.int64)
    _ensure_int32_range(acc, what="int32 linear output")
    return np.ascontiguousarray(acc.astype(np.int32))


def numpy_linear_int32(
    x: npt.NDArray[np.int32],
    w: npt.NDArray[np.int32],
    b: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    return np.ascontiguousarray(x.dot(w) + b, dtype=np.int32)


def numpy_relu_int32(x: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    return np.ascontiguousarray(np.maximum(x, np.int32(0)), dtype=np.int32)


def reference_mlp_int32(
    x: npt.NDArray[np.int32],
    w1: npt.NDArray[np.int32],
    b1: npt.NDArray[np.int32],
    w2: npt.NDArray[np.int32],
    b2: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    hidden = reference_linear_int32(x, w1, b1)
    hidden = numpy_relu_int32(hidden)
    return reference_linear_int32(hidden, w2, b2)


def numpy_mlp_int32(
    x: npt.NDArray[np.int32],
    w1: npt.NDArray[np.int32],
    b1: npt.NDArray[np.int32],
    w2: npt.NDArray[np.int32],
    b2: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    hidden = numpy_linear_int32(x, w1, b1)
    hidden = numpy_relu_int32(hidden)
    return numpy_linear_int32(hidden, w2, b2)


def torch_mlp_fp32(
    x: npt.NDArray[np.float32],
    w1: npt.NDArray[np.float32],
    b1: npt.NDArray[np.float32],
    w2: npt.NDArray[np.float32],
    b2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if torch is None or torch_f is None:
        raise RuntimeError("torch is not available")

    with torch.no_grad():
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        w1_t = torch.from_numpy(np.ascontiguousarray(w1.T))
        b1_t = torch.from_numpy(np.ascontiguousarray(b1))
        w2_t = torch.from_numpy(np.ascontiguousarray(w2.T))
        b2_t = torch.from_numpy(np.ascontiguousarray(b2))
        hidden = torch.relu(torch_f.linear(x_t, w1_t, b1_t))
        output = torch_f.linear(hidden, w2_t, b2_t)
    return output.cpu().numpy()


def torch_mlp_int32(
    x: npt.NDArray[np.int32],
    w1: npt.NDArray[np.int32],
    b1: npt.NDArray[np.int32],
    w2: npt.NDArray[np.int32],
    b2: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    if torch is None:
        raise RuntimeError("torch is not available")

    with torch.no_grad():
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        w1_t = torch.from_numpy(np.ascontiguousarray(w1))
        b1_t = torch.from_numpy(np.ascontiguousarray(b1))
        w2_t = torch.from_numpy(np.ascontiguousarray(w2))
        b2_t = torch.from_numpy(np.ascontiguousarray(b2))
        hidden = torch.clamp_min(x_t.matmul(w1_t) + b1_t, 0)
        output = hidden.matmul(w2_t) + b2_t
    return output.cpu().numpy()


@dataclass(frozen=True)
class PreparedInput:
    matrix: npt.NDArray[np.generic]
    batch: int


@dataclass(frozen=True)
class LinearDispatch:
    code: Array[np.uint64]
    uniforms: Array[np.uint32]
    thread: int
    wgs_per_sg: int
    workgroup: tuple[int, int, int] | None


@dataclass(frozen=True)
class BiasDispatch:
    code: Array[np.uint64]
    uniforms: Array[np.uint32]
    workgroup: tuple[int, int, int]
    thread: int


@dataclass(frozen=True)
class QpuBenchmarkStats:
    prep_sec: float
    cached_total_sec: float
    execute_only_sec: float
    max_abs_error: float


def _build_sgemm_dispatch(
    drv: Driver,
    code_payload: Array[np.uint64],
    code_qpu_aware: Array[np.uint64],
    a: Array[np.float32],
    b: Array[np.float32],
    c: Array[np.float32],
    *,
    q: int,
) -> LinearDispatch:
    p, q_a = a.shape
    q_b, r = b.shape
    if q_a != q_b or c.shape != (p, r):
        raise ValueError("invalid SGEMM buffer shapes")
    if p % P_TILE != 0 or q % Q_TILE != 0 or r % R_TILE != 0:
        raise ValueError("SGEMM buffers must already be padded to kernel tile sizes")

    tile_p = p // P_TILE
    tile_r_pair = r // R_TILE
    total_tasks = tile_p * tile_r_pair

    if total_tasks < VC7_QPUS:
        unif = drv.alloc(9, dtype=np.uint32)
        unif[0] = a.strides[0]
        unif[1] = a.addresses().item(0)
        unif[2] = b.strides[0]
        unif[3] = b.addresses().item(0)
        unif[4] = c.strides[0]
        unif[5] = c.addresses().item(0)
        unif[6] = q
        unif[7] = np.float32(1.0).view(np.uint32).item()
        unif[8] = np.float32(0.0).view(np.uint32).item()
        return LinearDispatch(
            code=code_payload,
            uniforms=unif,
            thread=total_tasks,
            wgs_per_sg=24,
            workgroup=(tile_r_pair, tile_p, 1),
        )

    task_counts = [0 for _ in range(VC7_QPUS)]
    for task_idx in range(total_tasks):
        task_counts[task_idx % VC7_QPUS] += 1

    max_tasks = max(task_counts)
    streams = drv.alloc((VC7_QPUS, max_tasks, 10), dtype=np.uint32)
    meta = drv.alloc((VC7_QPUS, 16), dtype=np.uint32)
    global_unif = drv.alloc(2, dtype=np.uint32)

    streams[:] = 0
    meta[:] = 0

    alpha_bits = np.float32(1.0).view(np.uint32).item()
    beta_bits = np.float32(0.0).view(np.uint32).item()

    next_slot = [0 for _ in range(VC7_QPUS)]
    task_idx = 0
    for tile_i in range(tile_p):
        for tile_j_pair in range(tile_r_pair):
            qpu_id = task_idx % VC7_QPUS
            slot = next_slot[qpu_id]
            record = streams[qpu_id, slot]
            record[0] = ((tile_i & 0xFFFF) << 16) | (tile_j_pair & 0xFFFF)
            record[1] = a.strides[0]
            record[2] = a.addresses().item(0)
            record[3] = b.strides[0]
            record[4] = b.addresses().item(0)
            record[5] = c.strides[0]
            record[6] = c.addresses().item(0)
            record[7] = q
            record[8] = alpha_bits
            record[9] = beta_bits
            next_slot[qpu_id] += 1
            task_idx += 1

    for qpu_id, count in enumerate(task_counts):
        if count == 0:
            continue
        meta[qpu_id, 0] = streams.addresses()[qpu_id, 0, 0]
        meta[qpu_id, 1] = count

    global_unif[0] = meta.addresses()[0, 0]
    global_unif[1] = meta.strides[0]

    return LinearDispatch(
        code=code_qpu_aware,
        uniforms=global_unif,
        thread=VC7_QPUS,
        wgs_per_sg=VC7_QPUS,
        workgroup=None,
    )


def _build_validated_tiled_matmul_dispatch(
    drv: Driver,
    code: Array[np.uint64],
    a: Array[np.generic],
    b: Array[np.generic],
    c: Array[np.generic],
    *,
    q: int,
) -> LinearDispatch:
    p, q_a = a.shape
    q_b, r = b.shape
    if q_a != q_b or c.shape != (p, r):
        raise ValueError("invalid tiled matmul buffer shapes")
    if p % P_TILE != 0 or q % Q_TILE != 0 or r % VALIDATED_LINEAR_R_TILE != 0:
        raise ValueError("tiled matmul buffers must already be padded to the validated kernel tile sizes")

    unif = drv.alloc(7, dtype=np.uint32)
    unif[0] = a.strides[0]
    unif[1] = a.addresses().item(0)
    unif[2] = b.strides[0]
    unif[3] = b.addresses().item(0)
    unif[4] = c.strides[0]
    unif[5] = c.addresses().item(0)
    unif[6] = q

    return LinearDispatch(
        code=code,
        uniforms=unif,
        thread=(p // P_TILE) * (r // VALIDATED_LINEAR_R_TILE),
        wgs_per_sg=24,
        workgroup=(r // VALIDATED_LINEAR_R_TILE, p // P_TILE, 1),
    )


def _build_igemm_dispatch(
    drv: Driver,
    code_payload: Array[np.uint64],
    code_qpu_aware: Array[np.uint64],
    a: Array[np.int32],
    b: Array[np.int32],
    c: Array[np.int32],
    *,
    q: int,
) -> LinearDispatch:
    p, q_a = a.shape
    q_b, r = b.shape
    if q_a != q_b or c.shape != (p, r):
        raise ValueError("invalid IGEMM buffer shapes")
    if p % P_TILE != 0 or q % Q_TILE != 0 or r % R_TILE != 0:
        raise ValueError("IGEMM buffers must already be padded to kernel tile sizes")

    tile_p = p // P_TILE
    tile_r_pair = r // R_TILE
    total_tasks = tile_p * tile_r_pair

    if total_tasks < VC7_QPUS:
        unif = drv.alloc(7, dtype=np.uint32)
        unif[0] = a.strides[0]
        unif[1] = a.addresses().item(0)
        unif[2] = b.strides[0]
        unif[3] = b.addresses().item(0)
        unif[4] = c.strides[0]
        unif[5] = c.addresses().item(0)
        unif[6] = q
        return LinearDispatch(
            code=code_payload,
            uniforms=unif,
            thread=total_tasks,
            wgs_per_sg=24,
            workgroup=(tile_r_pair, tile_p, 1),
        )

    task_counts = [0 for _ in range(VC7_QPUS)]
    for task_idx in range(total_tasks):
        task_counts[task_idx % VC7_QPUS] += 1

    max_tasks = max(task_counts)
    streams = drv.alloc((VC7_QPUS, max_tasks, 8), dtype=np.uint32)
    meta = drv.alloc((VC7_QPUS, 16), dtype=np.uint32)
    global_unif = drv.alloc(2, dtype=np.uint32)

    streams[:] = 0
    meta[:] = 0

    next_slot = [0 for _ in range(VC7_QPUS)]
    task_idx = 0
    for tile_i in range(tile_p):
        for tile_j_pair in range(tile_r_pair):
            qpu_id = task_idx % VC7_QPUS
            slot = next_slot[qpu_id]
            record = streams[qpu_id, slot]
            record[0] = ((tile_i & 0xFFFF) << 16) | (tile_j_pair & 0xFFFF)
            record[1] = a.strides[0]
            record[2] = a.addresses().item(0)
            record[3] = b.strides[0]
            record[4] = b.addresses().item(0)
            record[5] = c.strides[0]
            record[6] = c.addresses().item(0)
            record[7] = q
            next_slot[qpu_id] += 1
            task_idx += 1

    for qpu_id, count in enumerate(task_counts):
        if count == 0:
            continue
        meta[qpu_id, 0] = streams.addresses()[qpu_id, 0, 0]
        meta[qpu_id, 1] = count

    global_unif[0] = meta.addresses()[0, 0]
    global_unif[1] = meta.strides[0]

    return LinearDispatch(
        code=code_qpu_aware,
        uniforms=global_unif,
        thread=VC7_QPUS,
        wgs_per_sg=VC7_QPUS,
        workgroup=None,
    )


def _build_bias_dispatch(
    drv: Driver,
    code: Array[np.uint64],
    matrix: Array[np.generic],
    bias: Array[np.generic] | None,
) -> BiasDispatch:
    p, r = matrix.shape
    if p % P_TILE != 0 or r % 16 != 0:
        raise ValueError("invalid bias/relu matrix shape")

    uniforms = drv.alloc(3, dtype=np.uint32)
    uniforms[0] = matrix.strides[0]
    uniforms[1] = matrix.addresses().item(0)
    uniforms[2] = 0 if bias is None else bias.addresses().item(0)
    return BiasDispatch(
        code=code,
        uniforms=uniforms,
        workgroup=(r // 16, p // P_TILE, 1),
        thread=(p // P_TILE) * (r // 16),
    )


def _execute_linear_dispatch(drv: Driver, dispatch: LinearDispatch) -> None:
    kwargs = {
        "code": dispatch.code,
        "local_invocation": (16, 1, 1),
        "uniforms": dispatch.uniforms.addresses().item(0),
        "wgs_per_sg": dispatch.wgs_per_sg,
        "thread": dispatch.thread,
    }
    if dispatch.workgroup is not None:
        kwargs["workgroup"] = dispatch.workgroup
    drv.execute(**kwargs)


def _execute_bias_dispatch(drv: Driver, dispatch: BiasDispatch) -> None:
    drv.execute(
        dispatch.code,
        local_invocation=(16, 1, 1),
        uniforms=dispatch.uniforms.addresses().item(0),
        workgroup=dispatch.workgroup,
        wgs_per_sg=24,
        thread=dispatch.thread,
    )


@qpu
def qpu_igemm_rnn_reuse_a_x2(asm: Assembly) -> None:
    reg_tile_i = rf1
    reg_tile_j_pair = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b0 = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b0_base = rf14
    reg_c_stride = rf10
    reg_c0_base = rf15

    reg_accum0 = [rf[i] for i in range(16, 32)]
    reg_accum1 = [rf[i] for i in range(32, 48)]

    reg_b1 = [rf48, rf49, rf50, rf51]
    reg_b1_base = rf52
    reg_c1_base = rf53
    reg_tmp = rf57
    reg_a3_next = rf58
    reg_mul = rf59

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j_pair, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_a_stride))
    umul24(reg_tmp, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(reg_tmp, reg_tmp, 4)
    add(reg_a_base, reg_a_base, reg_tmp, sig=ldunifrf(reg_b_stride))
    shl(reg_tmp, reg_tile_j_pair, 7)
    nop(sig=ldunifrf(reg_b0_base))
    eidx(reg_tmp).add(reg_b0_base, reg_b0_base, reg_tmp)

    umul24(reg_mul, reg_tmp, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, reg_mul)
    shr(reg_mul, reg_tmp, 2)
    band(reg_tmp, reg_tmp, 3)
    shl(reg_tmp, reg_tmp, 4).umul24(reg_mul, reg_mul, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(reg_tmp, reg_tmp, reg_mul)
    del reg_b_stride
    add(reg_b0_base, reg_b0_base, reg_tmp)
    mov(reg_mul, 1)
    shl(reg_mul, reg_mul, 4)
    shl(reg_mul, reg_mul, 2)
    add(reg_b1_base, reg_b0_base, reg_mul)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)
    bnot(tmuc, 3)
    mov(tmua, reg_b0_base).add(reg_b0_base, reg_b0_base, reg_b_stride_x4)
    bnot(tmuc, 3)
    mov(tmua, reg_b1_base, sig=thrsw).add(reg_b1_base, reg_b1_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j_pair, 3).umul24(reg_tmp, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j_pair
    eidx(rf0).add(reg_tmp, reg_tmp, rf0, sig=ldunifrf(reg_c0_base))
    shl(reg_tmp, reg_tmp, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c0_base, reg_c0_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c0_base, reg_c0_base, reg_tmp)
    mov(reg_mul, 1)
    shl(reg_mul, reg_mul, 4)
    shl(reg_mul, reg_mul, 2)
    add(reg_c1_base, reg_c0_base, reg_mul)

    all_accum = reg_accum0 + reg_accum1
    initial_loads = reg_a + reg_b0 + reg_b1
    for i in range(16):
        r1 = all_accum[i]
        r2 = all_accum[i + 16]
        if i < len(initial_loads):
            bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu(initial_loads[i]))
        else:
            bxor(r1, r1, r1).sub(r2, r2, r2)

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)
        bnot(tmuc, 3)
        mov(tmua, reg_b0_base).add(reg_b0_base, reg_b0_base, reg_b_stride_x4)
        bnot(tmuc, 3)
        mov(tmua, reg_b1_base, sig=thrsw).add(reg_b1_base, reg_b1_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b0_order = deque(reg_b0 * 16)
        rotate_broadcast_reg_b1_order = deque(reg_b1 * 16)

        def rotate_broadcast_reg_b(tile: int) -> None:
            order = rotate_broadcast_reg_b0_order if tile == 0 else rotate_broadcast_reg_b1_order
            r = order.popleft()
            rotate(r, r, 1).mov(rep, r)

        states: list[tuple[int, Register, Register]] = []
        for reg_a_item in reg_a:
            for tile, reg_accum in enumerate((reg_accum0, reg_accum1)):
                for reg_accum_item in reg_accum:
                    states.append((tile, reg_accum_item, reg_a_item))

        reload_targets = [reg_a[0], reg_a[1], reg_a[2], reg_a3_next, *reg_b0, *reg_b1]
        reload_base_idx = len(states) - len(reload_targets)

        def load_sig(state_idx: int):
            if state_idx < reload_base_idx:
                return None
            return ldtmu(reload_targets[state_idx - reload_base_idx])

        rotate_broadcast_reg_b(states[0][0])
        sub(reg_i, reg_i, 1, cond="pushz").smul24(rf1, rf0, states[0][2])

        for idx in range(1, len(states) - 1):
            _, prev_accum, _ = states[idx - 1]
            cur_tile, _, cur_a = states[idx]
            rotate_broadcast_reg_b(cur_tile)
            sig = load_sig(idx - 1)
            if sig is None:
                add(prev_accum, prev_accum, rf1).smul24(rf1, rf0, cur_a)
            else:
                add(prev_accum, prev_accum, rf1, sig=sig).smul24(rf1, rf0, cur_a)

        lk.b(cond="anyna")
        rotate_broadcast_reg_b(states[-1][0])

        sig = load_sig(len(states) - 2)
        if sig is None:
            add(states[-2][1], states[-2][1], rf1).smul24(rf1, rf0, states[-1][2])
        else:
            add(states[-2][1], states[-2][1], rf1, sig=sig).smul24(rf1, rf0, states[-1][2])

        sig = load_sig(len(states) - 1)
        if sig is None:
            add(states[-1][1], states[-1][1], rf1).mov(reg_a[3], reg_a3_next)
        else:
            add(states[-1][1], states[-1][1], rf1, sig=sig).mov(reg_a[3], reg_a3_next)

    del reg_a
    del reg_b0
    del reg_b1
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b0_base
    del reg_b1_base

    def store_tile(reg_accum: list[Register], reg_c_base: Register) -> None:
        mov(tmuc, -1)
        for i in range(0, 16, 4):
            mov(tmud, reg_accum[i])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 1])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 2])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 3])
            mov(tmua, reg_c_base)
            tmuwt()
            if i < 12:
                sub(reg_c_base, reg_c_base, -4)

    store_tile(reg_accum0, reg_c0_base)
    store_tile(reg_accum1, reg_c1_base)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_igemm_rnn_reuse_a_x2_qpu_aware(asm: Assembly) -> None:
    reg_meta_base = rf60
    reg_stream_base = rf61
    reg_task_count = rf62
    reg_meta_stride = rf63

    tidx(rf0, sig=ldunifrf(reg_meta_base))
    shr(rf0, rf0, 2)
    band(rf0, rf0, 0b1111)

    nop(sig=ldunifrf(reg_meta_stride))
    umul24(rf1, rf0, reg_meta_stride)
    add(reg_meta_base, reg_meta_base, rf1)

    eidx(rf1)
    shl(rf1, rf1, 2)
    add(tmua, reg_meta_base, rf1, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(rf2))
    bcastf(reg_stream_base, rf2)
    rotate(rf2, rf2, 1)
    bcastf(reg_task_count, rf2)

    b(R.task_loop, cond="always").unif_addr(reg_stream_base)
    nop()
    nop()
    nop()

    L.task_loop

    nop(sig=ldunifrf(rf3))

    reg_tile_i = rf1
    reg_tile_j_pair = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b0 = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b0_base = rf14
    reg_c_stride = rf10
    reg_c0_base = rf15

    reg_accum0 = [rf[i] for i in range(16, 32)]
    reg_accum1 = [rf[i] for i in range(32, 48)]

    reg_b1 = [rf48, rf49, rf50, rf51]
    reg_b1_base = rf52
    reg_c1_base = rf53
    reg_tmp = rf57
    reg_a3_next = rf58
    reg_mul = rf59

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j_pair, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_a_stride))
    umul24(reg_tmp, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(reg_tmp, reg_tmp, 4)
    add(reg_a_base, reg_a_base, reg_tmp, sig=ldunifrf(reg_b_stride))
    shl(reg_tmp, reg_tile_j_pair, 7)
    nop(sig=ldunifrf(reg_b0_base))
    eidx(reg_tmp).add(reg_b0_base, reg_b0_base, reg_tmp)

    umul24(reg_mul, reg_tmp, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, reg_mul)
    shr(reg_mul, reg_tmp, 2)
    band(reg_tmp, reg_tmp, 3)
    shl(reg_tmp, reg_tmp, 4).umul24(reg_mul, reg_mul, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(reg_tmp, reg_tmp, reg_mul)
    del reg_b_stride
    add(reg_b0_base, reg_b0_base, reg_tmp)
    mov(reg_mul, 1)
    shl(reg_mul, reg_mul, 4)
    shl(reg_mul, reg_mul, 2)
    add(reg_b1_base, reg_b0_base, reg_mul)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)
    bnot(tmuc, 3)
    mov(tmua, reg_b0_base).add(reg_b0_base, reg_b0_base, reg_b_stride_x4)
    bnot(tmuc, 3)
    mov(tmua, reg_b1_base, sig=thrsw).add(reg_b1_base, reg_b1_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j_pair, 3).umul24(reg_tmp, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j_pair
    eidx(rf0).add(reg_tmp, reg_tmp, rf0, sig=ldunifrf(reg_c0_base))
    shl(reg_tmp, reg_tmp, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c0_base, reg_c0_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c0_base, reg_c0_base, reg_tmp)
    mov(reg_mul, 1)
    shl(reg_mul, reg_mul, 4)
    shl(reg_mul, reg_mul, 2)
    add(reg_c1_base, reg_c0_base, reg_mul)

    all_accum = reg_accum0 + reg_accum1
    initial_loads = reg_a + reg_b0 + reg_b1
    for i in range(16):
        r1 = all_accum[i]
        r2 = all_accum[i + 16]
        if i < len(initial_loads):
            bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu(initial_loads[i]))
        else:
            bxor(r1, r1, r1).sub(r2, r2, r2)

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)
        bnot(tmuc, 3)
        mov(tmua, reg_b0_base).add(reg_b0_base, reg_b0_base, reg_b_stride_x4)
        bnot(tmuc, 3)
        mov(tmua, reg_b1_base, sig=thrsw).add(reg_b1_base, reg_b1_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b0_order = deque(reg_b0 * 16)
        rotate_broadcast_reg_b1_order = deque(reg_b1 * 16)

        def rotate_broadcast_reg_b(tile: int) -> None:
            order = rotate_broadcast_reg_b0_order if tile == 0 else rotate_broadcast_reg_b1_order
            r = order.popleft()
            rotate(r, r, 1).mov(rep, r)

        states: list[tuple[int, Register, Register]] = []
        for reg_a_item in reg_a:
            for tile, reg_accum in enumerate((reg_accum0, reg_accum1)):
                for reg_accum_item in reg_accum:
                    states.append((tile, reg_accum_item, reg_a_item))

        reload_targets = [reg_a[0], reg_a[1], reg_a[2], reg_a3_next, *reg_b0, *reg_b1]
        reload_base_idx = len(states) - len(reload_targets)

        def load_sig(state_idx: int):
            if state_idx < reload_base_idx:
                return None
            return ldtmu(reload_targets[state_idx - reload_base_idx])

        rotate_broadcast_reg_b(states[0][0])
        sub(reg_i, reg_i, 1, cond="pushz").smul24(rf1, rf0, states[0][2])

        for idx in range(1, len(states) - 1):
            _, prev_accum, _ = states[idx - 1]
            cur_tile, _, cur_a = states[idx]
            rotate_broadcast_reg_b(cur_tile)
            sig = load_sig(idx - 1)
            if sig is None:
                add(prev_accum, prev_accum, rf1).smul24(rf1, rf0, cur_a)
            else:
                add(prev_accum, prev_accum, rf1, sig=sig).smul24(rf1, rf0, cur_a)

        lk.b(cond="anyna")
        rotate_broadcast_reg_b(states[-1][0])

        sig = load_sig(len(states) - 2)
        if sig is None:
            add(states[-2][1], states[-2][1], rf1).smul24(rf1, rf0, states[-1][2])
        else:
            add(states[-2][1], states[-2][1], rf1, sig=sig).smul24(rf1, rf0, states[-1][2])

        sig = load_sig(len(states) - 1)
        if sig is None:
            add(states[-1][1], states[-1][1], rf1).mov(reg_a[3], reg_a3_next)
        else:
            add(states[-1][1], states[-1][1], rf1, sig=sig).mov(reg_a[3], reg_a3_next)

    del reg_a
    del reg_b0
    del reg_b1
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b0_base
    del reg_b1_base

    def store_tile(reg_accum: list[Register], reg_c_base: Register) -> None:
        mov(tmuc, -1)
        for i in range(0, 16, 4):
            mov(tmud, reg_accum[i])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 1])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 2])
            mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -4)
            mov(tmud, reg_accum[i + 3])
            mov(tmua, reg_c_base)
            tmuwt()
            if i < 12:
                sub(reg_c_base, reg_c_base, -4)

    store_tile(reg_accum0, reg_c0_base)
    store_tile(reg_accum1, reg_c1_base)

    sub(reg_task_count, reg_task_count, 1, cond="pushz")
    b(R.task_loop, cond="anyna")
    nop()
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
def qpu_tiled_bias_activation_fp32(asm: Assembly, *, use_bias: bool, apply_relu: bool) -> None:
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_stride = rf4
    reg_base = rf5
    reg_bias_base = rf6
    reg_row_ptr = rf7
    reg_bias = rf8
    reg_val = rf9
    reg_out = rf10
    reg_zero = rf11
    reg_tmp = rf12

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_stride))
    umul24(reg_tmp, reg_tile_i, reg_stride, sig=ldunifrf(reg_base))
    shl(reg_tmp, reg_tmp, 4)
    if use_bias:
        add(reg_base, reg_base, reg_tmp, sig=ldunifrf(reg_bias_base))
    else:
        add(reg_base, reg_base, reg_tmp)

    shl(reg_tmp, reg_tile_j, 6)
    eidx(rf0)
    shl(rf0, rf0, 2)
    add(reg_row_ptr, reg_base, reg_tmp)
    add(reg_row_ptr, reg_row_ptr, rf0)

    mov(reg_zero, 0.0)
    if use_bias:
        add(reg_bias_base, reg_bias_base, reg_tmp)
        add(reg_bias_base, reg_bias_base, rf0)
        mov(tmua, reg_bias_base, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_bias))
    else:
        mov(reg_bias, 0.0)

    mov(tmuc, -1)
    for i in range(16):
        mov(tmua, reg_row_ptr, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_val))
        if use_bias:
            fadd(reg_out, reg_val, reg_bias)
        else:
            mov(reg_out, reg_val)
        if apply_relu:
            fmax(reg_out, reg_out, reg_zero)
        mov(tmud, reg_out)
        mov(tmua, reg_row_ptr)
        tmuwt()
        if i < 15:
            add(reg_row_ptr, reg_row_ptr, reg_stride)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_tiled_bias_activation_int32(asm: Assembly, *, use_bias: bool, apply_relu: bool) -> None:
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_stride = rf4
    reg_base = rf5
    reg_bias_base = rf6
    reg_row_ptr = rf7
    reg_bias = rf8
    reg_val = rf9
    reg_out = rf10
    reg_zero = rf11
    reg_tmp = rf12

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_stride))
    umul24(reg_tmp, reg_tile_i, reg_stride, sig=ldunifrf(reg_base))
    shl(reg_tmp, reg_tmp, 4)
    if use_bias:
        add(reg_base, reg_base, reg_tmp, sig=ldunifrf(reg_bias_base))
    else:
        add(reg_base, reg_base, reg_tmp)

    shl(reg_tmp, reg_tile_j, 6)
    eidx(rf0)
    shl(rf0, rf0, 2)
    add(reg_row_ptr, reg_base, reg_tmp)
    add(reg_row_ptr, reg_row_ptr, rf0)

    mov(reg_zero, 0)
    if use_bias:
        add(reg_bias_base, reg_bias_base, reg_tmp)
        add(reg_bias_base, reg_bias_base, rf0)
        mov(tmua, reg_bias_base, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_bias))
    else:
        mov(reg_bias, 0)

    mov(tmuc, -1)
    for i in range(16):
        mov(tmua, reg_row_ptr, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(reg_val))
        if use_bias:
            add(reg_out, reg_val, reg_bias)
        else:
            mov(reg_out, reg_val)
        if apply_relu:
            imax(reg_out, reg_out, reg_zero)
        mov(tmud, reg_out)
        mov(tmua, reg_row_ptr)
        tmuwt()
        if i < 15:
            add(reg_row_ptr, reg_row_ptr, reg_stride)

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


class _BaseTiledMlpExecutor:
    batch: int
    in_features: int
    hidden_features: int
    out_features: int
    batch_padded: int
    in_padded: int
    hidden_padded: int
    out_padded: int
    _drv: Driver
    _input_dev: Array[np.generic]
    _hidden_dev: Array[np.generic]
    _output_dev: Array[np.generic]
    _linear1_dispatch: LinearDispatch
    _linear2_dispatch: LinearDispatch
    _hidden_bias_relu_dispatch: BiasDispatch
    _hidden_bias_only_dispatch: BiasDispatch
    _relu_only_dispatch: BiasDispatch
    _output_bias_only_dispatch: BiasDispatch

    def close(self) -> None:
        self._drv.close()

    def __enter__(self) -> "_BaseTiledMlpExecutor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _prepare_matrix(
        self,
        x: npt.NDArray[np.generic],
        *,
        expected_shape: tuple[int, int],
        padded_shape: tuple[int, int],
        dtype: np.dtype[np.generic],
    ) -> PreparedInput:
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {x.shape}")
        if x.dtype != dtype:
            raise ValueError(f"expected dtype {dtype}, got {x.dtype}")
        if x.shape != expected_shape:
            raise ValueError(f"expected shape {expected_shape}, got {x.shape}")
        matrix = np.zeros(padded_shape, dtype=dtype)
        matrix[: expected_shape[0], : expected_shape[1]] = x
        return PreparedInput(matrix=np.ascontiguousarray(matrix), batch=expected_shape[0])

    def prepare_input(self, x: npt.NDArray[np.generic]) -> PreparedInput:
        return self._prepare_matrix(
            x,
            expected_shape=(self.batch, self.in_features),
            padded_shape=(self.batch_padded, self.in_padded),
            dtype=self._input_dev.dtype,
        )

    def prepare_hidden(self, hidden: npt.NDArray[np.generic]) -> PreparedInput:
        return self._prepare_matrix(
            hidden,
            expected_shape=(self.batch, self.hidden_features),
            padded_shape=(self.batch_padded, self.hidden_padded),
            dtype=self._hidden_dev.dtype,
        )

    def upload_input(self, prepared: PreparedInput) -> None:
        self._input_dev[:] = prepared.matrix

    def upload_hidden(self, prepared: PreparedInput) -> None:
        self._hidden_dev[:] = prepared.matrix

    def execute_linear1(self) -> None:
        _execute_linear_dispatch(self._drv, self._linear1_dispatch)

    def execute_hidden_bias_relu(self) -> None:
        _execute_bias_dispatch(self._drv, self._hidden_bias_relu_dispatch)

    def execute_hidden_bias_only(self) -> None:
        _execute_bias_dispatch(self._drv, self._hidden_bias_only_dispatch)

    def execute_relu_only(self) -> None:
        _execute_bias_dispatch(self._drv, self._relu_only_dispatch)

    def execute_linear2(self) -> None:
        _execute_linear_dispatch(self._drv, self._linear2_dispatch)

    def execute_output_bias_only(self) -> None:
        _execute_bias_dispatch(self._drv, self._output_bias_only_dispatch)

    def execute_mlp(self) -> None:
        self.execute_linear1()
        self.execute_hidden_bias_relu()
        self.execute_linear2()
        self.execute_output_bias_only()

    def read_hidden(self) -> npt.NDArray[np.generic]:
        hidden = np.array(self._hidden_dev, copy=True)
        return np.ascontiguousarray(hidden[: self.batch, : self.hidden_features])

    def read_output(self) -> npt.NDArray[np.generic]:
        output = np.array(self._output_dev, copy=True)
        return np.ascontiguousarray(output[: self.batch, : self.out_features])


class TiledMlpExecutor(_BaseTiledMlpExecutor):
    def __init__(
        self,
        w1: npt.NDArray[np.float32],
        b1: npt.NDArray[np.float32],
        w2: npt.NDArray[np.float32],
        b2: npt.NDArray[np.float32],
        *,
        batch: int,
    ) -> None:
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("weights must be 2-D")
        if b1.ndim != 1 or b2.ndim != 1:
            raise ValueError("biases must be 1-D")
        if w1.dtype != np.float32 or w2.dtype != np.float32 or b1.dtype != np.float32 or b2.dtype != np.float32:
            raise ValueError("fp32 MLP expects float32 weights and biases")
        if w1.shape[1] != b1.shape[0]:
            raise ValueError("w1 and b1 shapes do not match")
        if w2.shape[0] != w1.shape[1]:
            raise ValueError("w1 output features and w2 input features do not match")
        if w2.shape[1] != b2.shape[0]:
            raise ValueError("w2 and b2 shapes do not match")

        self.batch = int(batch)
        self.in_features = int(w1.shape[0])
        self.hidden_features = int(w1.shape[1])
        self.out_features = int(w2.shape[1])
        self.batch_padded = _round_up(self.batch, P_TILE)
        self.in_padded = _round_up(self.in_features, Q_TILE)
        self.hidden_padded = _round_up(self.hidden_features, VALIDATED_LINEAR_R_TILE)
        self.out_padded = _round_up(self.out_features, VALIDATED_LINEAR_R_TILE)

        data_area_size = (
            self.batch_padded * self.in_padded
            + self.in_padded * self.hidden_padded
            + self.batch_padded * self.hidden_padded
            + self.hidden_padded
            + self.hidden_padded * self.out_padded
            + self.batch_padded * self.out_padded
            + self.out_padded
        ) * np.dtype(np.float32).itemsize + (1 << 20)

        self._drv = Driver(data_area_size=data_area_size)
        tiled_matmul_code = self._drv.program(qpu_tiledconv2d_fp32)
        bias_relu_code = self._drv.program(qpu_tiled_bias_activation_fp32, use_bias=True, apply_relu=True)
        bias_only_code = self._drv.program(qpu_tiled_bias_activation_fp32, use_bias=True, apply_relu=False)
        relu_only_code = self._drv.program(qpu_tiled_bias_activation_fp32, use_bias=False, apply_relu=True)

        self._input_dev = self._drv.alloc((self.batch_padded, self.in_padded), dtype=np.float32)
        w1_dev = self._drv.alloc((self.in_padded, self.hidden_padded), dtype=np.float32)
        self._hidden_dev = self._drv.alloc((self.batch_padded, self.hidden_padded), dtype=np.float32)
        b1_dev = self._drv.alloc(self.hidden_padded, dtype=np.float32)
        w2_dev = self._drv.alloc((self.hidden_padded, self.out_padded), dtype=np.float32)
        self._output_dev = self._drv.alloc((self.batch_padded, self.out_padded), dtype=np.float32)
        b2_dev = self._drv.alloc(self.out_padded, dtype=np.float32)

        w1_dev[:] = self._pad_weight(w1, rows=self.in_padded, cols=self.hidden_padded, dtype=np.float32)
        w2_dev[:] = self._pad_weight(w2, rows=self.hidden_padded, cols=self.out_padded, dtype=np.float32)
        b1_dev[:] = self._pad_bias(b1, size=self.hidden_padded, dtype=np.float32)
        b2_dev[:] = self._pad_bias(b2, size=self.out_padded, dtype=np.float32)
        self._hidden_dev[:] = 0.0
        self._output_dev[:] = 0.0

        self._linear1_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._input_dev,
            w1_dev,
            self._hidden_dev,
            q=self.in_padded,
        )
        self._linear2_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._hidden_dev,
            w2_dev,
            self._output_dev,
            q=self.hidden_padded,
        )
        self._hidden_bias_relu_dispatch = _build_bias_dispatch(self._drv, bias_relu_code, self._hidden_dev, b1_dev)
        self._hidden_bias_only_dispatch = _build_bias_dispatch(self._drv, bias_only_code, self._hidden_dev, b1_dev)
        self._relu_only_dispatch = _build_bias_dispatch(self._drv, relu_only_code, self._hidden_dev, None)
        self._output_bias_only_dispatch = _build_bias_dispatch(self._drv, bias_only_code, self._output_dev, b2_dev)

    @staticmethod
    def _pad_weight(
        weight: npt.NDArray[np.float32],
        *,
        rows: int,
        cols: int,
        dtype: np.dtype[np.float32],
    ) -> npt.NDArray[np.float32]:
        padded = np.zeros((rows, cols), dtype=dtype)
        padded[: weight.shape[0], : weight.shape[1]] = weight
        return np.ascontiguousarray(padded)

    @staticmethod
    def _pad_bias(
        bias: npt.NDArray[np.float32],
        *,
        size: int,
        dtype: np.dtype[np.float32],
    ) -> npt.NDArray[np.float32]:
        padded = np.zeros(size, dtype=dtype)
        padded[: bias.shape[0]] = bias
        return np.ascontiguousarray(padded)


class TiledMlpExecutorInt32(_BaseTiledMlpExecutor):
    def __init__(
        self,
        w1: npt.NDArray[np.int32],
        b1: npt.NDArray[np.int32],
        w2: npt.NDArray[np.int32],
        b2: npt.NDArray[np.int32],
        *,
        batch: int,
        input_bound: int | None = None,
    ) -> None:
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("weights must be 2-D")
        if b1.ndim != 1 or b2.ndim != 1:
            raise ValueError("biases must be 1-D")
        if w1.dtype != np.int32 or w2.dtype != np.int32 or b1.dtype != np.int32 or b2.dtype != np.int32:
            raise ValueError("int32 MLP expects int32 weights and biases")
        if w1.shape[1] != b1.shape[0]:
            raise ValueError("w1 and b1 shapes do not match")
        if w2.shape[0] != w1.shape[1]:
            raise ValueError("w1 output features and w2 input features do not match")
        if w2.shape[1] != b2.shape[0]:
            raise ValueError("w2 and b2 shapes do not match")
        if input_bound is None:
            raise ValueError("int32 MLP executor requires an input bound for the hidden-range contract")

        dummy_input = np.full((1, w1.shape[0]), input_bound, dtype=np.int32)
        _validate_int32_mlp_contract(dummy_input, w1, b1, w2, b2)

        self.batch = int(batch)
        self.in_features = int(w1.shape[0])
        self.hidden_features = int(w1.shape[1])
        self.out_features = int(w2.shape[1])
        self.batch_padded = _round_up(self.batch, P_TILE)
        self.in_padded = _round_up(self.in_features, Q_TILE)
        self.hidden_padded = _round_up(self.hidden_features, VALIDATED_LINEAR_R_TILE)
        self.out_padded = _round_up(self.out_features, VALIDATED_LINEAR_R_TILE)

        data_area_size = (
            self.batch_padded * self.in_padded
            + self.in_padded * self.hidden_padded
            + self.batch_padded * self.hidden_padded
            + self.hidden_padded
            + self.hidden_padded * self.out_padded
            + self.batch_padded * self.out_padded
            + self.out_padded
        ) * np.dtype(np.int32).itemsize + (1 << 20)

        self._drv = Driver(data_area_size=data_area_size)
        tiled_matmul_code = self._drv.program(qpu_tiledconv2d_int32)
        bias_relu_code = self._drv.program(qpu_tiled_bias_activation_int32, use_bias=True, apply_relu=True)
        bias_only_code = self._drv.program(qpu_tiled_bias_activation_int32, use_bias=True, apply_relu=False)
        relu_only_code = self._drv.program(qpu_tiled_bias_activation_int32, use_bias=False, apply_relu=True)

        self._input_dev = self._drv.alloc((self.batch_padded, self.in_padded), dtype=np.int32)
        w1_dev = self._drv.alloc((self.in_padded, self.hidden_padded), dtype=np.int32)
        self._hidden_dev = self._drv.alloc((self.batch_padded, self.hidden_padded), dtype=np.int32)
        b1_dev = self._drv.alloc(self.hidden_padded, dtype=np.int32)
        w2_dev = self._drv.alloc((self.hidden_padded, self.out_padded), dtype=np.int32)
        self._output_dev = self._drv.alloc((self.batch_padded, self.out_padded), dtype=np.int32)
        b2_dev = self._drv.alloc(self.out_padded, dtype=np.int32)

        w1_dev[:] = self._pad_weight(w1, rows=self.in_padded, cols=self.hidden_padded, dtype=np.int32)
        w2_dev[:] = self._pad_weight(w2, rows=self.hidden_padded, cols=self.out_padded, dtype=np.int32)
        b1_dev[:] = self._pad_bias(b1, size=self.hidden_padded, dtype=np.int32)
        b2_dev[:] = self._pad_bias(b2, size=self.out_padded, dtype=np.int32)
        self._hidden_dev[:] = 0
        self._output_dev[:] = 0

        self._linear1_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._input_dev,
            w1_dev,
            self._hidden_dev,
            q=self.in_padded,
        )
        self._linear2_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._hidden_dev,
            w2_dev,
            self._output_dev,
            q=self.hidden_padded,
        )
        self._hidden_bias_relu_dispatch = _build_bias_dispatch(self._drv, bias_relu_code, self._hidden_dev, b1_dev)
        self._hidden_bias_only_dispatch = _build_bias_dispatch(self._drv, bias_only_code, self._hidden_dev, b1_dev)
        self._relu_only_dispatch = _build_bias_dispatch(self._drv, relu_only_code, self._hidden_dev, None)
        self._output_bias_only_dispatch = _build_bias_dispatch(self._drv, bias_only_code, self._output_dev, b2_dev)

    @staticmethod
    def _pad_weight(
        weight: npt.NDArray[np.int32],
        *,
        rows: int,
        cols: int,
        dtype: np.dtype[np.int32],
    ) -> npt.NDArray[np.int32]:
        padded = np.zeros((rows, cols), dtype=dtype)
        padded[: weight.shape[0], : weight.shape[1]] = weight
        return np.ascontiguousarray(padded)

    @staticmethod
    def _pad_bias(
        bias: npt.NDArray[np.int32],
        *,
        size: int,
        dtype: np.dtype[np.int32],
    ) -> npt.NDArray[np.int32]:
        padded = np.zeros(size, dtype=dtype)
        padded[: bias.shape[0]] = bias
        return np.ascontiguousarray(padded)


def tiledmlp_fp32(
    x: npt.NDArray[np.float32],
    w1: npt.NDArray[np.float32],
    b1: npt.NDArray[np.float32],
    w2: npt.NDArray[np.float32],
    b2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    with TiledMlpExecutor(w1, b1, w2, b2, batch=int(x.shape[0])) as executor:
        prepared = executor.prepare_input(x)
        executor.upload_input(prepared)
        executor.execute_mlp()
        return executor.read_output()


def tiledmlp_int32(
    x: npt.NDArray[np.int32],
    w1: npt.NDArray[np.int32],
    b1: npt.NDArray[np.int32],
    w2: npt.NDArray[np.int32],
    b2: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    _validate_int32_mlp_contract(x, w1, b1, w2, b2)
    with TiledMlpExecutorInt32(w1, b1, w2, b2, batch=int(x.shape[0]), input_bound=_max_abs_int(x)) as executor:
        prepared = executor.prepare_input(x)
        executor.upload_input(prepared)
        executor.execute_mlp()
        return executor.read_output()


def _benchmark_qpu_operator(
    prepare_fn,
    upload_fn,
    execute_fn,
    read_fn,
) -> tuple[npt.NDArray[np.generic], QpuBenchmarkStats]:
    prepared, prep_sec = _benchmark_callable(prepare_fn, repeat=5)

    def run_total():
        upload_fn(prepared)
        execute_fn()
        return read_fn()

    result, cached_total_sec = _benchmark_callable(run_total, repeat=5)
    upload_fn(prepared)
    execute_only_sec = _benchmark_timing(execute_fn, repeat=10)
    diff = np.abs(result.astype(np.float64))
    _ = diff
    return result, QpuBenchmarkStats(
        prep_sec=prep_sec,
        cached_total_sec=cached_total_sec,
        execute_only_sec=execute_only_sec,
        max_abs_error=0.0,
    )


def _with_error(
    stats: QpuBenchmarkStats,
    *,
    actual: npt.NDArray[np.generic],
    expected: npt.NDArray[np.generic],
) -> QpuBenchmarkStats:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return QpuBenchmarkStats(
        prep_sec=stats.prep_sec,
        cached_total_sec=stats.cached_total_sec,
        execute_only_sec=stats.execute_only_sec,
        max_abs_error=float(np.max(diff)),
    )


def _print_qpu_stats(
    title: str,
    *,
    numpy_sec: float,
    torch_sec: float | None,
    throughput_fn,
    qpu_stats: QpuBenchmarkStats | None,
    torch_label: str,
) -> None:
    print(f"-- {title} --")
    print(f"numpy: {numpy_sec:.4f} sec, {throughput_fn(numpy_sec):.4f} Gop/s")
    if torch_sec is None:
        print(f"{torch_label}: n/a")
    else:
        print(f"{torch_label}: {torch_sec:.4f} sec, {throughput_fn(torch_sec):.4f} Gop/s")
    if qpu_stats is None:
        print("QPU: unavailable")
    else:
        prep_cached_sec = qpu_stats.prep_sec + qpu_stats.cached_total_sec
        print(f"QPU host prep: {qpu_stats.prep_sec:.4f} sec")
        print(f"QPU cached total: {qpu_stats.cached_total_sec:.4f} sec, {throughput_fn(qpu_stats.cached_total_sec):.4f} Gop/s")
        print(f"QPU execute only: {qpu_stats.execute_only_sec:.4f} sec, {throughput_fn(qpu_stats.execute_only_sec):.4f} Gop/s")
        print(f"QPU prep+cached total: {prep_cached_sec:.4f} sec, {throughput_fn(prep_cached_sec):.4f} Gop/s")
        print(f"Maximum absolute error: {qpu_stats.max_abs_error}")
    print()


def benchmark_tiledmlp_fp32() -> dict[str, float]:
    batch = 256
    in_features = 1024
    hidden_features = 1024
    out_features = 512

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(batch, in_features)).astype(np.float32)
    w1 = rng.uniform(-1.0, 1.0, size=(in_features, hidden_features)).astype(np.float32)
    b1 = rng.uniform(-0.5, 0.5, size=(hidden_features,)).astype(np.float32)
    w2 = rng.uniform(-1.0, 1.0, size=(hidden_features, out_features)).astype(np.float32)
    b2 = rng.uniform(-0.5, 0.5, size=(out_features,)).astype(np.float32)

    expected_layer1, numpy_layer1_sec = _benchmark_callable(lambda: numpy_linear_fp32(x, w1, b1), repeat=3)
    expected_relu, numpy_relu_sec = _benchmark_callable(lambda: numpy_relu_fp32(expected_layer1), repeat=5)
    expected_layer2, numpy_layer2_sec = _benchmark_callable(lambda: numpy_linear_fp32(expected_relu, w2, b2), repeat=3)
    expected_total, numpy_total_sec = _benchmark_callable(lambda: numpy_mlp_naive(x, w1, b1, w2, b2), repeat=3)
    assert np.allclose(expected_total, expected_layer2, atol=1e-5, rtol=1e-5)

    torch_layer1_sec = None
    torch_relu_sec = None
    torch_layer2_sec = None
    torch_total_sec = None
    if torch is not None and torch_f is not None:
        with torch.no_grad():
            x_t = torch.from_numpy(np.ascontiguousarray(x))
            w1_t = torch.from_numpy(np.ascontiguousarray(w1.T))
            b1_t = torch.from_numpy(np.ascontiguousarray(b1))
            w2_t = torch.from_numpy(np.ascontiguousarray(w2.T))
            b2_t = torch.from_numpy(np.ascontiguousarray(b2))
            hidden_t = torch.from_numpy(np.ascontiguousarray(expected_layer1))
            relu_t = torch.from_numpy(np.ascontiguousarray(expected_relu))

            def run_torch_layer1():
                return torch_f.linear(x_t, w1_t, b1_t)

            def run_torch_relu():
                return torch.relu(hidden_t)

            def run_torch_layer2():
                return torch_f.linear(relu_t, w2_t, b2_t)

            def run_torch_total():
                hidden = torch.relu(torch_f.linear(x_t, w1_t, b1_t))
                return torch_f.linear(hidden, w2_t, b2_t)

            assert np.allclose(run_torch_layer1().cpu().numpy(), expected_layer1, atol=5e-4, rtol=5e-4)
            assert np.allclose(run_torch_relu().cpu().numpy(), expected_relu, atol=0.0, rtol=0.0)
            assert np.allclose(run_torch_layer2().cpu().numpy(), expected_layer2, atol=5e-4, rtol=5e-4)
            assert np.allclose(run_torch_total().cpu().numpy(), expected_total, atol=5e-4, rtol=5e-4)

            torch_layer1_sec = _benchmark_timing(run_torch_layer1, repeat=5)
            torch_relu_sec = _benchmark_timing(run_torch_relu, repeat=5)
            torch_layer2_sec = _benchmark_timing(run_torch_layer2, repeat=5)
            torch_total_sec = _benchmark_timing(run_torch_total, repeat=5)

    print("==== tiledmlp fp32 example ====")
    print("Operator: Linear -> ReLU -> Linear")
    print(f"Dimensions: x={batch}x{in_features}, w1={in_features}x{hidden_features}, w2={hidden_features}x{out_features}")
    print("Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.")

    result: dict[str, float] = {"numpy_sec": numpy_total_sec}
    setup_start = getsec()
    try:
        executor = TiledMlpExecutor(w1, b1, w2, b2, batch=batch)
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        print()
        _print_qpu_stats(
            "MLP total",
            numpy_sec=numpy_total_sec,
            torch_sec=torch_total_sec,
            throughput_fn=lambda sec: mlp_gops(batch, in_features, hidden_features, out_features, sec),
            qpu_stats=None,
            torch_label="torch native mlp",
        )
        _print_qpu_stats(
            "Layer1 only",
            numpy_sec=numpy_layer1_sec,
            torch_sec=torch_layer1_sec,
            throughput_fn=lambda sec: linear_gops(batch, in_features, hidden_features, sec),
            qpu_stats=None,
            torch_label="torch native linear",
        )
        _print_qpu_stats(
            "ReLU only",
            numpy_sec=numpy_relu_sec,
            torch_sec=torch_relu_sec,
            throughput_fn=lambda sec: relu_gops(batch, hidden_features, sec),
            qpu_stats=None,
            torch_label="torch relu",
        )
        _print_qpu_stats(
            "Layer2 only",
            numpy_sec=numpy_layer2_sec,
            torch_sec=torch_layer2_sec,
            throughput_fn=lambda sec: linear_gops(batch, hidden_features, out_features, sec),
            qpu_stats=None,
            torch_label="torch native linear",
        )
        if torch_total_sec is not None:
            result["torch_sec"] = torch_total_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        total_actual, total_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_input(x),
            executor.upload_input,
            executor.execute_mlp,
            executor.read_output,
        )
        total_stats = _with_error(total_stats, actual=total_actual, expected=expected_total)

        layer1_actual, layer1_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_input(x),
            executor.upload_input,
            lambda: (executor.execute_linear1(), executor.execute_hidden_bias_only()),
            executor.read_hidden,
        )
        layer1_stats = _with_error(layer1_stats, actual=layer1_actual, expected=expected_layer1)

        relu_actual, relu_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_hidden(expected_layer1),
            executor.upload_hidden,
            executor.execute_relu_only,
            executor.read_hidden,
        )
        relu_stats = _with_error(relu_stats, actual=relu_actual, expected=expected_relu)

        layer2_actual, layer2_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_hidden(expected_relu),
            executor.upload_hidden,
            lambda: (executor.execute_linear2(), executor.execute_output_bias_only()),
            executor.read_output,
        )
        layer2_stats = _with_error(layer2_stats, actual=layer2_actual, expected=expected_layer2)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    print()
    _print_qpu_stats(
        "MLP total",
        numpy_sec=numpy_total_sec,
        torch_sec=torch_total_sec,
        throughput_fn=lambda sec: mlp_gops(batch, in_features, hidden_features, out_features, sec),
        qpu_stats=total_stats,
        torch_label="torch native mlp",
    )
    _print_qpu_stats(
        "Layer1 only",
        numpy_sec=numpy_layer1_sec,
        torch_sec=torch_layer1_sec,
        throughput_fn=lambda sec: linear_gops(batch, in_features, hidden_features, sec),
        qpu_stats=layer1_stats,
        torch_label="torch native linear",
    )
    _print_qpu_stats(
        "ReLU only",
        numpy_sec=numpy_relu_sec,
        torch_sec=torch_relu_sec,
        throughput_fn=lambda sec: relu_gops(batch, hidden_features, sec),
        qpu_stats=relu_stats,
        torch_label="torch relu",
    )
    _print_qpu_stats(
        "Layer2 only",
        numpy_sec=numpy_layer2_sec,
        torch_sec=torch_layer2_sec,
        throughput_fn=lambda sec: linear_gops(batch, hidden_features, out_features, sec),
        qpu_stats=layer2_stats,
        torch_label="torch native linear",
    )

    result.update(
        {
            "torch_sec": -1.0 if torch_total_sec is None else torch_total_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": total_stats.prep_sec,
            "qpu_cached_total_sec": total_stats.cached_total_sec,
            "qpu_execute_only_sec": total_stats.execute_only_sec,
            "max_abs_error": total_stats.max_abs_error,
        }
    )
    return result


def benchmark_tiledmlp_int32() -> dict[str, float]:
    batch = 256
    in_features = 1024
    hidden_features = 1024
    out_features = 512

    rng = np.random.default_rng(1)
    x = rng.integers(-8, 8, size=(batch, in_features), dtype=np.int32)
    w1 = rng.integers(-8, 8, size=(in_features, hidden_features), dtype=np.int32)
    b1 = rng.integers(-128, 128, size=(hidden_features,), dtype=np.int32)
    w2 = rng.integers(-8, 8, size=(hidden_features, out_features), dtype=np.int32)
    b2 = rng.integers(-128, 128, size=(out_features,), dtype=np.int32)

    _validate_int32_mlp_contract(x, w1, b1, w2, b2)

    expected_layer1 = reference_linear_int32(x, w1, b1)
    expected_relu = numpy_relu_int32(expected_layer1)
    expected_layer2 = reference_linear_int32(expected_relu, w2, b2)
    expected_total = reference_mlp_int32(x, w1, b1, w2, b2)
    np.testing.assert_array_equal(expected_total, expected_layer2)

    numpy_layer1_actual, numpy_layer1_sec = _benchmark_callable(lambda: numpy_linear_int32(x, w1, b1), repeat=3)
    numpy_relu_actual, numpy_relu_sec = _benchmark_callable(lambda: numpy_relu_int32(expected_layer1), repeat=5)
    numpy_layer2_actual, numpy_layer2_sec = _benchmark_callable(lambda: numpy_linear_int32(expected_relu, w2, b2), repeat=3)
    numpy_total_actual, numpy_total_sec = _benchmark_callable(lambda: numpy_mlp_int32(x, w1, b1, w2, b2), repeat=3)
    np.testing.assert_array_equal(numpy_layer1_actual, expected_layer1)
    np.testing.assert_array_equal(numpy_relu_actual, expected_relu)
    np.testing.assert_array_equal(numpy_layer2_actual, expected_layer2)
    np.testing.assert_array_equal(numpy_total_actual, expected_total)

    torch_layer1_sec = None
    torch_relu_sec = None
    torch_layer2_sec = None
    torch_total_sec = None
    torch_error: str | None = None
    if torch is not None:
        try:
            with torch.no_grad():
                x_t = torch.from_numpy(np.ascontiguousarray(x))
                w1_t = torch.from_numpy(np.ascontiguousarray(w1))
                b1_t = torch.from_numpy(np.ascontiguousarray(b1))
                w2_t = torch.from_numpy(np.ascontiguousarray(w2))
                b2_t = torch.from_numpy(np.ascontiguousarray(b2))
                hidden_t = torch.from_numpy(np.ascontiguousarray(expected_layer1))
                relu_t = torch.from_numpy(np.ascontiguousarray(expected_relu))

                def run_torch_layer1():
                    return x_t.matmul(w1_t) + b1_t

                def run_torch_relu():
                    return torch.clamp_min(hidden_t, 0)

                def run_torch_layer2():
                    return relu_t.matmul(w2_t) + b2_t

                def run_torch_total():
                    hidden = torch.clamp_min(x_t.matmul(w1_t) + b1_t, 0)
                    return hidden.matmul(w2_t) + b2_t

                np.testing.assert_array_equal(run_torch_layer1().cpu().numpy(), expected_layer1)
                np.testing.assert_array_equal(run_torch_relu().cpu().numpy(), expected_relu)
                np.testing.assert_array_equal(run_torch_layer2().cpu().numpy(), expected_layer2)
                np.testing.assert_array_equal(run_torch_total().cpu().numpy(), expected_total)

                torch_layer1_sec = _benchmark_timing(run_torch_layer1, repeat=5)
                torch_relu_sec = _benchmark_timing(run_torch_relu, repeat=5)
                torch_layer2_sec = _benchmark_timing(run_torch_layer2, repeat=5)
                torch_total_sec = _benchmark_timing(run_torch_total, repeat=5)
        except RuntimeError as exc:
            torch_error = str(exc)

    print("==== tiledmlp int32 example ====")
    print("Operator: Linear -> ReLU -> Linear")
    print(f"Dimensions: x={batch}x{in_features}, w1={in_features}x{hidden_features}, w2={hidden_features}x{out_features}")
    print("Kernel contract: x, w1, and w2 must fit the signed 24-bit range, and layer1 output must also stay within it.")
    print("Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.")
    if torch_error is not None:
        print(f"Torch int32 path unavailable: {torch_error}")

    result: dict[str, float] = {"numpy_sec": numpy_total_sec}
    setup_start = getsec()
    try:
        executor = TiledMlpExecutorInt32(w1, b1, w2, b2, batch=batch, input_bound=_max_abs_int(x))
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        print()
        _print_qpu_stats(
            "MLP total",
            numpy_sec=numpy_total_sec,
            torch_sec=torch_total_sec,
            throughput_fn=lambda sec: mlp_gops(batch, in_features, hidden_features, out_features, sec),
            qpu_stats=None,
            torch_label="torch int32 mlp",
        )
        _print_qpu_stats(
            "Layer1 only",
            numpy_sec=numpy_layer1_sec,
            torch_sec=torch_layer1_sec,
            throughput_fn=lambda sec: linear_gops(batch, in_features, hidden_features, sec),
            qpu_stats=None,
            torch_label="torch int32 linear",
        )
        _print_qpu_stats(
            "ReLU only",
            numpy_sec=numpy_relu_sec,
            torch_sec=torch_relu_sec,
            throughput_fn=lambda sec: relu_gops(batch, hidden_features, sec),
            qpu_stats=None,
            torch_label="torch int32 relu",
        )
        _print_qpu_stats(
            "Layer2 only",
            numpy_sec=numpy_layer2_sec,
            torch_sec=torch_layer2_sec,
            throughput_fn=lambda sec: linear_gops(batch, hidden_features, out_features, sec),
            qpu_stats=None,
            torch_label="torch int32 linear",
        )
        if torch_total_sec is not None:
            result["torch_sec"] = torch_total_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        total_actual, total_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_input(x),
            executor.upload_input,
            executor.execute_mlp,
            executor.read_output,
        )
        total_stats = _with_error(total_stats, actual=total_actual, expected=expected_total)

        layer1_actual, layer1_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_input(x),
            executor.upload_input,
            lambda: (executor.execute_linear1(), executor.execute_hidden_bias_only()),
            executor.read_hidden,
        )
        layer1_stats = _with_error(layer1_stats, actual=layer1_actual, expected=expected_layer1)

        relu_actual, relu_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_hidden(expected_layer1),
            executor.upload_hidden,
            executor.execute_relu_only,
            executor.read_hidden,
        )
        relu_stats = _with_error(relu_stats, actual=relu_actual, expected=expected_relu)

        layer2_actual, layer2_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_hidden(expected_relu),
            executor.upload_hidden,
            lambda: (executor.execute_linear2(), executor.execute_output_bias_only()),
            executor.read_output,
        )
        layer2_stats = _with_error(layer2_stats, actual=layer2_actual, expected=expected_layer2)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    print()
    _print_qpu_stats(
        "MLP total",
        numpy_sec=numpy_total_sec,
        torch_sec=torch_total_sec,
        throughput_fn=lambda sec: mlp_gops(batch, in_features, hidden_features, out_features, sec),
        qpu_stats=total_stats,
        torch_label="torch int32 mlp",
    )
    _print_qpu_stats(
        "Layer1 only",
        numpy_sec=numpy_layer1_sec,
        torch_sec=torch_layer1_sec,
        throughput_fn=lambda sec: linear_gops(batch, in_features, hidden_features, sec),
        qpu_stats=layer1_stats,
        torch_label="torch int32 linear",
    )
    _print_qpu_stats(
        "ReLU only",
        numpy_sec=numpy_relu_sec,
        torch_sec=torch_relu_sec,
        throughput_fn=lambda sec: relu_gops(batch, hidden_features, sec),
        qpu_stats=relu_stats,
        torch_label="torch int32 relu",
    )
    _print_qpu_stats(
        "Layer2 only",
        numpy_sec=numpy_layer2_sec,
        torch_sec=torch_layer2_sec,
        throughput_fn=lambda sec: linear_gops(batch, hidden_features, out_features, sec),
        qpu_stats=layer2_stats,
        torch_label="torch int32 linear",
    )

    result.update(
        {
            "torch_sec": -1.0 if torch_total_sec is None else torch_total_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": total_stats.prep_sec,
            "qpu_cached_total_sec": total_stats.cached_total_sec,
            "qpu_execute_only_sec": total_stats.execute_only_sec,
            "max_abs_error": total_stats.max_abs_error,
        }
    )
    return result


def main() -> None:
    benchmark_tiledmlp_fp32()
    print()
    benchmark_tiledmlp_int32()


if __name__ == "__main__":
    main()
