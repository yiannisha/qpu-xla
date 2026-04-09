from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import CLOCK_MONOTONIC, clock_gettime
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError:
    torch = None

from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def conv2d_gops(
    batch: int,
    in_channels: int,
    out_channels: int,
    out_height: int,
    out_width: int,
    kernel_height: int,
    kernel_width: int,
    sec: float,
) -> float:
    macs = batch * out_channels * out_height * out_width * in_channels * kernel_height * kernel_width
    return (2 * macs) / sec * 1e-9


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        assert len(value) == 2
        return value
    return (value, value)


def _round_up(value: int, tile: int) -> int:
    return ((value + tile - 1) // tile) * tile


def _conv2d_output_hw(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    eff_kernel_h = dil_h * (kernel_height - 1) + 1
    eff_kernel_w = dil_w * (kernel_width - 1) + 1
    out_height = (height + 2 * pad_h - eff_kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - eff_kernel_w) // stride_w + 1
    if out_height <= 0 or out_width <= 0:
        raise ValueError("invalid conv2d geometry: output dimensions must be positive")
    return out_height, out_width


def im2col_nchw(
    x: npt.NDArray[np.generic],
    kernel_size: tuple[int, int],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.generic]:
    if x.ndim != 4:
        raise ValueError(f"expected NCHW input, got shape {x.shape}")

    kernel_height, kernel_width = kernel_size
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    batch, channels, height, width = x.shape

    out_height, out_width = _conv2d_output_hw(
        height,
        width,
        kernel_height,
        kernel_width,
        (stride_h, stride_w),
        (pad_h, pad_w),
        (dil_h, dil_w),
    )

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    cols = np.empty((batch, out_height, out_width, channels, kernel_height, kernel_width), dtype=x.dtype)

    for ky in range(kernel_height):
        y = slice(ky * dil_h, ky * dil_h + out_height * stride_h, stride_h)
        for kx in range(kernel_width):
            x_idx = slice(kx * dil_w, kx * dil_w + out_width * stride_w, stride_w)
            patch = x_padded[:, :, y, x_idx]
            cols[:, :, :, :, ky, kx] = patch.transpose(0, 2, 3, 1)

    return np.ascontiguousarray(cols.reshape(batch * out_height * out_width, channels * kernel_height * kernel_width))


def reference_conv2d_nchw(
    x: npt.NDArray[np.generic],
    weight: npt.NDArray[np.generic],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.generic]:
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("expected NCHW input and OIHW weights")

    batch, in_channels, height, width = x.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if weight_channels != in_channels:
        raise ValueError("input and weight channel counts do not match")

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    out_height, out_width = _conv2d_output_hw(
        height,
        width,
        kernel_height,
        kernel_width,
        (stride_h, stride_w),
        (pad_h, pad_w),
        (dil_h, dil_w),
    )

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    output = np.zeros((batch, out_channels, out_height, out_width), dtype=np.result_type(x.dtype, weight.dtype))

    for n in range(batch):
        for oc in range(out_channels):
            for oy in range(out_height):
                iy = oy * stride_h
                for ox in range(out_width):
                    ix = ox * stride_w
                    acc = output[n, oc, oy, ox]
                    for ic in range(in_channels):
                        for ky in range(kernel_height):
                            py = iy + ky * dil_h
                            for kx in range(kernel_width):
                                px = ix + kx * dil_w
                                acc = acc + x_padded[n, ic, py, px] * weight[oc, ic, ky, kx]
                    output[n, oc, oy, ox] = acc

    return output


def pack_int16_pairs(a: npt.NDArray[np.int16]) -> npt.NDArray[np.int32]:
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    if a.shape[1] % 2 != 0:
        raise ValueError("the packed dimension must be even")

    lo = a[:, 0::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    hi = a[:, 1::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    packed = lo | (hi << 16)
    return np.ascontiguousarray(packed.view(np.int32))


def _reshape_weight_oihw_to_gemm(weight: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    matrix = weight.transpose(1, 2, 3, 0).reshape(
        in_channels * kernel_height * kernel_width,
        out_channels,
    )
    return np.ascontiguousarray(matrix)


def _pad_matrix_rows_cols(
    a: npt.NDArray[np.generic],
    b: npt.NDArray[np.generic],
    *,
    p_tile: int,
    q_tile: int,
    r_tile: int,
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic], int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matrix shapes do not align")

    p, q = a.shape
    _, r = b.shape
    p_padded = _round_up(p, p_tile)
    q_padded = _round_up(q, q_tile)
    r_padded = _round_up(r, r_tile)

    a_padded = np.zeros((p_padded, q_padded), dtype=a.dtype)
    b_padded = np.zeros((q_padded, r_padded), dtype=b.dtype)
    a_padded[:p, :q] = a
    b_padded[:q, :r] = b

    return (
        np.ascontiguousarray(a_padded),
        np.ascontiguousarray(b_padded),
        p,
        q,
        r,
    )


def _emit_packed_mac_step(
    reg_accum: list[Register],
    reg_a_unpacked: Register,
    reg_b_unpacked: Register,
    reg_i: Register,
    a_reg: Register,
    a_half: str,
    b_reg: Register,
    b_half: str,
    *,
    first: bool = False,
) -> None:
    mov(reg_a_unpacked, a_reg.unpack(a_half))
    mov(reg_b_unpacked, b_reg.unpack(b_half))

    rotate(reg_b_unpacked, reg_b_unpacked, 1).mov(rep, reg_b_unpacked)
    if first:
        sub(reg_i, reg_i, 1, cond="pushz").smul24(rf1, rf0, reg_a_unpacked)
    else:
        nop().smul24(rf1, rf0, reg_a_unpacked)

    for i in range(15):
        rotate(reg_b_unpacked, reg_b_unpacked, 1).mov(rep, reg_b_unpacked)
        add(reg_accum[i], reg_accum[i], rf1).smul24(rf1, rf0, reg_a_unpacked)

    add(reg_accum[15], reg_accum[15], rf1)


def _reshape_gemm_output(
    c: npt.NDArray[np.generic],
    output_shape: tuple[int, int, int, int],
    p: int,
    r: int,
) -> npt.NDArray[np.generic]:
    n, out_channels, out_height, out_width = output_shape
    matrix = c[:p, :r].reshape(n, out_height, out_width, out_channels)
    return np.ascontiguousarray(matrix.transpose(0, 3, 1, 2))


def _conv_gops_from_output(
    output: npt.NDArray[np.generic],
    in_channels: int,
    kernel_height: int,
    kernel_width: int,
    sec: float,
) -> float:
    return conv2d_gops(
        batch=int(output.shape[0]),
        in_channels=in_channels,
        out_channels=int(output.shape[1]),
        out_height=int(output.shape[2]),
        out_width=int(output.shape[3]),
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        sec=sec,
    )


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
    fn: Callable[[], None],
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


def numpy_conv2d_nchw(
    x: npt.NDArray[np.generic],
    weight: npt.NDArray[np.generic],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    compute_dtype: npt.DTypeLike | None = None,
    out_dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.generic]:
    a, b, output_shape = _prepare_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    if compute_dtype is not None:
        a = a.astype(compute_dtype, copy=False)
        b = b.astype(compute_dtype, copy=False)

    c = a.dot(b)
    if out_dtype is not None:
        c = c.astype(out_dtype, copy=False)
    return _reshape_gemm_output(c, output_shape, a.shape[0], b.shape[1])


def torch_conv2d_nchw(
    x: npt.NDArray[np.generic],
    weight: npt.NDArray[np.generic],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.generic]:
    if torch is None:
        raise RuntimeError("torch is not available")

    with torch.no_grad():
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        w_t = torch.from_numpy(np.ascontiguousarray(weight))
        y_t = torch.nn.functional.conv2d(
            x_t,
            w_t,
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
        )
    return y_t.cpu().numpy()


@dataclass
class PreparedConvProblem:
    a: npt.NDArray[np.generic]
    b: npt.NDArray[np.generic]
    output_shape: tuple[int, int, int, int]
    p: int
    r: int


class TiledMatmulExecutor:
    _drv: Driver
    _code: Array[np.uint64]
    _a_dev: Array[np.generic]
    _b_dev: Array[np.generic]
    _c_dev: Array[np.generic]
    _unif: Array[np.uint32]
    _workgroup: tuple[int, int, int]
    _thread: int

    def __init__(
        self,
        qpu_kernel: Callable[[Assembly], Any],
        a_shape: tuple[int, int],
        a_dtype: npt.DTypeLike,
        b_shape: tuple[int, int],
        b_dtype: npt.DTypeLike,
        out_dtype: npt.DTypeLike,
    ) -> None:
        a_dtype_np = np.dtype(a_dtype)
        b_dtype_np = np.dtype(b_dtype)
        out_dtype_np = np.dtype(out_dtype)
        p, q = a_shape
        q_b, r = b_shape
        if q != q_b:
            raise ValueError("matrix shapes do not align")

        data_area_size = (
            p * q * a_dtype_np.itemsize
            + q_b * r * b_dtype_np.itemsize
            + p * r * out_dtype_np.itemsize
            + 4096
        )
        self._drv = Driver(data_area_size=data_area_size)
        self._code = self._drv.program(qpu_kernel)
        self._a_dev = self._drv.alloc(a_shape, dtype=a_dtype_np)
        self._b_dev = self._drv.alloc(b_shape, dtype=b_dtype_np)
        self._c_dev = self._drv.alloc((p, r), dtype=out_dtype_np)
        self._unif = self._drv.alloc(7, dtype=np.uint32)
        self._unif[0] = self._a_dev.strides[0]
        self._unif[1] = self._a_dev.addresses().item(0)
        self._unif[2] = self._b_dev.strides[0]
        self._unif[3] = self._b_dev.addresses().item(0)
        self._unif[4] = self._c_dev.strides[0]
        self._unif[5] = self._c_dev.addresses().item(0)
        self._unif[6] = q

        tile_p = p // 16
        tile_r = r // 16
        self._workgroup = (tile_r, tile_p, 1)
        self._thread = tile_p * tile_r

    def close(self) -> None:
        self._drv.close()

    def __enter__(self) -> "TiledMatmulExecutor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def upload(self, a: npt.NDArray[np.generic], b: npt.NDArray[np.generic]) -> None:
        self._a_dev[:] = a
        self._b_dev[:] = b

    def execute(self) -> None:
        self._drv.execute(
            self._code,
            local_invocation=(16, 1, 1),
            uniforms=self._unif.addresses().item(0),
            workgroup=self._workgroup,
            wgs_per_sg=24,
            thread=self._thread,
        )

    def readback(self) -> npt.NDArray[np.generic]:
        return np.array(self._c_dev, copy=True)


@qpu
def qpu_tiledconv2d_fp32(asm: Assembly) -> None:
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_a_stride))
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)

    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)
    del reg_b_stride
    add(reg_b_base, reg_b_base, rf3)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    del reg_c_stride
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b_order = deque(reg_b * 16)

        def rotate_broadcast_reg_b() -> None:
            r = rotate_broadcast_reg_b_order.popleft()
            rotate(r, r, 1).mov(rep, r)

        rotate_broadcast_reg_b()
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, reg_a[0])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[0])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[1])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[1])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[2])
        for i in range(15):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[2])
        rotate_broadcast_reg_b()
        fadd(reg_accum[15], reg_accum[15], rf1).fmul(rf1, rf0, reg_a[3])
        for i in range(8):
            rotate_broadcast_reg_b()
            fadd(reg_accum[i], reg_accum[i], rf1).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).fmul(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        fadd(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).fmul(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        rotate_broadcast_reg_b()
        fadd(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).fmul(rf1, rf0, reg_a[3])
        fadd(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base

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

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_tiledconv2d_int32(asm: Assembly) -> None:
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_a_stride))
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)

    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)
    del reg_b_stride
    add(reg_b_base, reg_b_base, rf3)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        rotate_broadcast_reg_b_order = deque(reg_b * 16)

        def rotate_broadcast_reg_b() -> None:
            r = rotate_broadcast_reg_b_order.popleft()
            rotate(r, r, 1).mov(rep, r)

        rotate_broadcast_reg_b()
        sub(reg_i, reg_i, 1, cond="pushz").smul24(rf1, rf0, reg_a[0])
        for i in range(15):
            rotate_broadcast_reg_b()
            add(reg_accum[i], reg_accum[i], rf1).smul24(rf1, rf0, reg_a[0])
        rotate_broadcast_reg_b()
        add(reg_accum[15], reg_accum[15], rf1).smul24(rf1, rf0, reg_a[1])
        for i in range(15):
            rotate_broadcast_reg_b()
            add(reg_accum[i], reg_accum[i], rf1).smul24(rf1, rf0, reg_a[1])
        rotate_broadcast_reg_b()
        add(reg_accum[15], reg_accum[15], rf1).smul24(rf1, rf0, reg_a[2])
        for i in range(15):
            rotate_broadcast_reg_b()
            add(reg_accum[i], reg_accum[i], rf1).smul24(rf1, rf0, reg_a[2])
        rotate_broadcast_reg_b()
        add(reg_accum[15], reg_accum[15], rf1).smul24(rf1, rf0, reg_a[3])
        for i in range(8):
            rotate_broadcast_reg_b()
            add(reg_accum[i], reg_accum[i], rf1).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[8], reg_accum[8], rf1, sig=ldtmu(reg_a[0])).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[9], reg_accum[9], rf1, sig=ldtmu(reg_a[1])).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[10], reg_accum[10], rf1, sig=ldtmu(reg_a[2])).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[11], reg_accum[11], rf1, sig=ldtmu(rf2)).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[12], reg_accum[12], rf1, sig=ldtmu(reg_b[0])).smul24(rf1, rf0, reg_a[3])
        rotate_broadcast_reg_b()
        add(reg_accum[13], reg_accum[13], rf1, sig=ldtmu(reg_b[1])).smul24(rf1, rf0, reg_a[3])
        lk.b(cond="anyna")
        rotate_broadcast_reg_b()
        add(reg_accum[14], reg_accum[14], rf1, sig=ldtmu(reg_b[2])).smul24(rf1, rf0, reg_a[3])
        add(reg_accum[15], reg_accum[15], rf1, sig=ldtmu(reg_b[3])).mov(reg_a[3], rf2)

    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base

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

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def qpu_tiledconv2d_int16_packed(asm: Assembly) -> None:
    reg_tile_i = rf1
    reg_tile_j = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b_base = rf14
    reg_c_stride = rf10
    reg_c_base = rf15
    reg_accum = [rf[i] for i in range(16, 32)]
    reg_a_unpacked = rf32
    reg_b_unpacked = rf33

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j, rf3.unpack("ul"))

    nop(sig=ldunifrf(reg_a_stride))
    umul24(rf3, reg_tile_i, reg_a_stride, sig=ldunifrf(reg_a_base))
    shl(rf3, rf3, 4)
    add(reg_a_base, reg_a_base, rf3, sig=ldunifrf(reg_b_stride))
    shl(rf3, reg_tile_j, 6)
    nop(sig=ldunifrf(reg_b_base))
    eidx(rf3).add(reg_b_base, reg_b_base, rf3)

    umul24(rf4, rf3, reg_a_stride)
    del reg_a_stride
    add(reg_a_base, reg_a_base, rf4)
    shr(rf4, rf3, 2)
    band(rf3, rf3, 3)
    shl(rf3, rf3, 4).umul24(rf4, rf4, reg_b_stride)
    shl(reg_b_stride_x4, reg_b_stride, 2).add(rf3, rf3, rf4)
    del reg_b_stride
    add(reg_b_base, reg_b_base, rf3)

    bnot(tmuc, 3)
    mov(tmua, reg_a_base)

    bnot(tmuc, 3)
    mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    nop(sig=ldunifrf(reg_c_stride))
    shl(rf0, reg_tile_j, 2).umul24(rf3, reg_tile_i, reg_c_stride)
    del reg_tile_i
    del reg_tile_j
    eidx(rf0).add(rf3, rf3, rf0, sig=ldunifrf(reg_c_base))
    shl(rf3, rf3, 4).umul24(rf0, rf0, reg_c_stride)
    add(reg_c_base, reg_c_base, rf0, sig=ldunif)
    shr(reg_i, rf0, 2).add(reg_c_base, reg_c_base, rf3)

    for i in range(8):
        r1 = reg_accum[i]
        r2 = reg_accum[i + 8]
        bxor(r1, r1, r1).sub(r2, r2, r2, sig=ldtmu((reg_a + reg_b)[i]))

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        _emit_packed_mac_step(
            reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[0], "il", reg_b[0], "il", first=True
        )
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[0], "ih", reg_b[0], "ih")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[1], "il", reg_b[1], "il")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[1], "ih", reg_b[1], "ih")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[2], "il", reg_b[2], "il")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[2], "ih", reg_b[2], "ih")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[3], "il", reg_b[3], "il")
        _emit_packed_mac_step(reg_accum, reg_a_unpacked, reg_b_unpacked, reg_i, reg_a[3], "ih", reg_b[3], "ih")

        for reg in reg_a + reg_b:
            nop(sig=ldtmu(reg))

        lk.b(cond="anyna")
        nop()
        nop()
        nop()

    del reg_a
    del reg_b
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b_base
    del reg_a_unpacked
    del reg_b_unpacked

    for i in range(16):
        mov(tmud, reg_accum[i])
        mov(tmua, reg_c_base)
        if i < 15:
            tmuwt().sub(reg_c_base, reg_c_base, -4)
        else:
            tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def _execute_tiled_matmul(
    qpu_kernel: Callable[[Assembly], Any],
    a: npt.NDArray[np.generic],
    b: npt.NDArray[np.generic],
    *,
    out_dtype: np.dtype[np.generic],
) -> npt.NDArray[np.generic]:
    with TiledMatmulExecutor(
        qpu_kernel,
        a.shape,
        a.dtype,
        b.shape,
        b.dtype,
        out_dtype,
    ) as executor:
        executor.upload(a, b)
        executor.execute()
        return executor.readback()


def _prepare_conv_problem(
    x: npt.NDArray[np.generic],
    weight: npt.NDArray[np.generic],
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic], tuple[int, int, int, int]]:
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("expected NCHW input and OIHW weights")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("input and weight channel counts do not match")

    out_channels = weight.shape[0]
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]
    out_height, out_width = _conv2d_output_hw(
        x.shape[2],
        x.shape[3],
        kernel_height,
        kernel_width,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
    )
    a = im2col_nchw(x, (kernel_height, kernel_width), stride=stride, padding=padding, dilation=dilation)
    b = _reshape_weight_oihw_to_gemm(weight)
    return a, b, (x.shape[0], out_channels, out_height, out_width)


def _prepare_fp32_conv_problem(
    x: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> PreparedConvProblem:
    a, b, output_shape = _prepare_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    a_padded, b_padded, p, _, r = _pad_matrix_rows_cols(a, b, p_tile=16, q_tile=4, r_tile=16)
    return PreparedConvProblem(a=a_padded, b=b_padded, output_shape=output_shape, p=p, r=r)


def _prepare_int32_conv_problem(
    x: npt.NDArray[np.int32],
    weight: npt.NDArray[np.int32],
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> PreparedConvProblem:
    a, b, output_shape = _prepare_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    a_padded, b_padded, p, _, r = _pad_matrix_rows_cols(a, b, p_tile=16, q_tile=4, r_tile=16)
    return PreparedConvProblem(a=a_padded, b=b_padded, output_shape=output_shape, p=p, r=r)


def _prepare_int16_conv_problem(
    x: npt.NDArray[np.int16],
    weight: npt.NDArray[np.int16],
    *,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> PreparedConvProblem:
    a, b, output_shape = _prepare_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    a_int16 = np.ascontiguousarray(a.astype(np.int16, copy=False))
    b_int16 = np.ascontiguousarray(b.astype(np.int16, copy=False))
    a_padded, b_padded, p, _, r = _pad_matrix_rows_cols(a_int16, b_int16, p_tile=16, q_tile=8, r_tile=16)
    a_packed = pack_int16_pairs(a_padded)
    b_packed = pack_int16_pairs(b_padded.T).T.copy()
    return PreparedConvProblem(a=a_packed, b=b_packed, output_shape=output_shape, p=p, r=r)


def _benchmark_prepare_problem(
    fn: Callable[[], PreparedConvProblem],
    *,
    repeat: int = 3,
) -> tuple[PreparedConvProblem, float]:
    prepared = fn()
    prep_sec = _benchmark_timing(lambda: fn(), repeat=repeat)
    return prepared, prep_sec


def _benchmark_cached_qpu_problem(
    executor: TiledMatmulExecutor,
    prepared: PreparedConvProblem,
) -> tuple[npt.NDArray[np.generic], float, float]:
    def run_total() -> npt.NDArray[np.generic]:
        executor.upload(prepared.a, prepared.b)
        executor.execute()
        return executor.readback()

    result, cached_total_sec = _benchmark_callable(run_total, repeat=5)
    executor.upload(prepared.a, prepared.b)
    execute_only_sec = _benchmark_timing(executor.execute, repeat=10)
    return result, cached_total_sec, execute_only_sec


def _print_qpu_benchmark(
    expected: npt.NDArray[np.generic],
    *,
    in_channels: int,
    kernel_height: int,
    kernel_width: int,
    prep_sec: float,
    cached_total_sec: float,
    execute_only_sec: float,
) -> None:
    cached_total_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, cached_total_sec)
    execute_only_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, execute_only_sec)
    end_to_end_cached_sec = prep_sec + cached_total_sec
    end_to_end_cached_gops = _conv_gops_from_output(
        expected,
        in_channels,
        kernel_height,
        kernel_width,
        end_to_end_cached_sec,
    )
    print(f"QPU host prep: {prep_sec:.4f} sec")
    print(f"QPU cached total: {cached_total_sec:.4f} sec, {cached_total_gops:.4f} Gop/s")
    print(f"QPU execute only: {execute_only_sec:.4f} sec, {execute_only_gops:.4f} Gop/s")
    print(f"QPU prep+cached total: {end_to_end_cached_sec:.4f} sec, {end_to_end_cached_gops:.4f} Gop/s")


def tiledconv2d_fp32(
    x: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.float32]:
    if x.dtype != np.float32 or weight.dtype != np.float32:
        raise ValueError("fp32 conv expects float32 inputs")

    prepared = _prepare_fp32_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    c = _execute_tiled_matmul(qpu_tiledconv2d_fp32, prepared.a, prepared.b, out_dtype=np.dtype(np.float32))
    return _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)


def tiledconv2d_int32(
    x: npt.NDArray[np.int32],
    weight: npt.NDArray[np.int32],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.int32]:
    if x.dtype != np.int32 or weight.dtype != np.int32:
        raise ValueError("int32 conv expects int32 inputs")

    max_input = int(max(np.max(np.abs(x), initial=0), np.max(np.abs(weight), initial=0)))
    if max_input >= (1 << 23):
        raise ValueError("int32 kernel uses smul24; all input values must fit the signed 24-bit range")

    prepared = _prepare_int32_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    c = _execute_tiled_matmul(qpu_tiledconv2d_int32, prepared.a, prepared.b, out_dtype=np.dtype(np.int32))
    return _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)


def tiledconv2d_int16(
    x: npt.NDArray[np.int16],
    weight: npt.NDArray[np.int16],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> npt.NDArray[np.int32]:
    if x.dtype != np.int16 or weight.dtype != np.int16:
        raise ValueError("int16 conv expects int16 inputs")

    prepared = _prepare_int16_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    c = _execute_tiled_matmul(qpu_tiledconv2d_int16_packed, prepared.a, prepared.b, out_dtype=np.dtype(np.int32))
    return _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)


def benchmark_tiledconv2d_fp32() -> dict[str, float]:
    batch, in_channels, height, width = 1, 32, 34, 34
    out_channels, kernel_height, kernel_width = 64, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    np.random.seed(0)
    x = np.random.uniform(-1.0, 1.0, size=(batch, in_channels, height, width)).astype(np.float32)
    weight = np.random.uniform(
        -1.0,
        1.0,
        size=(out_channels, in_channels, kernel_height, kernel_width),
    ).astype(np.float32)

    expected, time_numpy = _benchmark_callable(
        lambda: numpy_conv2d_nchw(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            compute_dtype=np.float32,
            out_dtype=np.float32,
        )
    )
    prepared, prep_sec = _benchmark_prepare_problem(
        lambda: _prepare_fp32_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    )
    with TiledMatmulExecutor(
        qpu_tiledconv2d_fp32,
        prepared.a.shape,
        prepared.a.dtype,
        prepared.b.shape,
        prepared.b.dtype,
        np.float32,
    ) as executor:
        c, cached_total_sec, execute_only_sec = _benchmark_cached_qpu_problem(executor, prepared)
    actual = _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)

    torch_sec = None
    if torch is not None:
        torch_output, torch_sec = _benchmark_callable(
            lambda: torch_conv2d_nchw(x, weight, stride=stride, padding=padding, dilation=dilation)
        )
        assert np.allclose(torch_output, expected, atol=1e-4, rtol=1e-4)

    abs_diff = np.abs(actual - expected)
    numpy_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, time_numpy)
    print("==== tiledconv2d fp32 example ====")
    print(f"numpy: {time_numpy:.4f} sec, {numpy_gops:.4f} Gop/s")
    if torch_sec is not None:
        torch_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, torch_sec)
        print(f"torch native conv2d: {torch_sec:.4f} sec, {torch_gops:.4f} Gop/s")
    _print_qpu_benchmark(
        expected,
        in_channels=in_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        prep_sec=prep_sec,
        cached_total_sec=cached_total_sec,
        execute_only_sec=execute_only_sec,
    )
    print(f"Maximum absolute error: {float(np.max(abs_diff))}")

    result = {
        "numpy_sec": time_numpy,
        "qpu_prep_sec": prep_sec,
        "qpu_cached_total_sec": cached_total_sec,
        "qpu_execute_only_sec": execute_only_sec,
        "max_abs_error": float(np.max(abs_diff)),
    }
    if torch_sec is not None:
        result["torch_sec"] = torch_sec
    return result


def benchmark_tiledconv2d_int32() -> dict[str, float]:
    batch, in_channels, height, width = 1, 32, 18, 18
    out_channels, kernel_height, kernel_width = 64, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    np.random.seed(0)
    x = np.random.randint(-32, 32, size=(batch, in_channels, height, width), dtype=np.int32)
    weight = np.random.randint(-32, 32, size=(out_channels, in_channels, kernel_height, kernel_width), dtype=np.int32)

    expected, time_numpy = _benchmark_callable(
        lambda: numpy_conv2d_nchw(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            compute_dtype=np.int32,
            out_dtype=np.int32,
        )
    )
    prepared, prep_sec = _benchmark_prepare_problem(
        lambda: _prepare_int32_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    )
    with TiledMatmulExecutor(
        qpu_tiledconv2d_int32,
        prepared.a.shape,
        prepared.a.dtype,
        prepared.b.shape,
        prepared.b.dtype,
        np.int32,
    ) as executor:
        c, cached_total_sec, execute_only_sec = _benchmark_cached_qpu_problem(executor, prepared)
    actual = _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)

    torch_sec = None
    if torch is not None:
        torch_output, torch_sec = _benchmark_callable(
            lambda: torch_conv2d_nchw(x, weight, stride=stride, padding=padding, dilation=dilation)
        )
        np.testing.assert_array_equal(torch_output, expected)

    diff = actual.astype(np.int64) - expected.astype(np.int64)
    numpy_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, time_numpy)
    print("==== tiledconv2d int32 example ====")
    print(f"numpy: {time_numpy:.4f} sec, {numpy_gops:.4f} Gop/s")
    if torch_sec is not None:
        torch_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, torch_sec)
        print(f"torch native conv2d: {torch_sec:.4f} sec, {torch_gops:.4f} Gop/s")
    _print_qpu_benchmark(
        expected,
        in_channels=in_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        prep_sec=prep_sec,
        cached_total_sec=cached_total_sec,
        execute_only_sec=execute_only_sec,
    )
    print(f"Maximum absolute error: {int(np.max(np.abs(diff)))}")

    result = {
        "numpy_sec": time_numpy,
        "qpu_prep_sec": prep_sec,
        "qpu_cached_total_sec": cached_total_sec,
        "qpu_execute_only_sec": execute_only_sec,
        "max_abs_error": float(np.max(np.abs(diff))),
    }
    if torch_sec is not None:
        result["torch_sec"] = torch_sec
    return result


def benchmark_tiledconv2d_int16() -> dict[str, float]:
    batch, in_channels, height, width = 1, 32, 18, 18
    out_channels, kernel_height, kernel_width = 64, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    np.random.seed(0)
    x = np.random.randint(-64, 64, size=(batch, in_channels, height, width), dtype=np.int16)
    weight = np.random.randint(-64, 64, size=(out_channels, in_channels, kernel_height, kernel_width), dtype=np.int16)

    expected, time_numpy = _benchmark_callable(
        lambda: numpy_conv2d_nchw(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            compute_dtype=np.int32,
            out_dtype=np.int32,
        )
    )
    prepared, prep_sec = _benchmark_prepare_problem(
        lambda: _prepare_int16_conv_problem(x, weight, stride=stride, padding=padding, dilation=dilation)
    )
    with TiledMatmulExecutor(
        qpu_tiledconv2d_int16_packed,
        prepared.a.shape,
        prepared.a.dtype,
        prepared.b.shape,
        prepared.b.dtype,
        np.int32,
    ) as executor:
        c, cached_total_sec, execute_only_sec = _benchmark_cached_qpu_problem(executor, prepared)
    actual = _reshape_gemm_output(c, prepared.output_shape, prepared.p, prepared.r)

    torch_sec = None
    if torch is not None:
        _, torch_sec = _benchmark_callable(
            lambda: torch_conv2d_nchw(x, weight, stride=stride, padding=padding, dilation=dilation)
        )

    diff = actual.astype(np.int64) - expected.astype(np.int64)
    numpy_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, time_numpy)
    print("==== tiledconv2d int16 example ====")
    print("Kernel contract: int16 inputs are packed as int32 pairs and accumulated into int32.")
    print(f"numpy: {time_numpy:.4f} sec, {numpy_gops:.4f} Gop/s")
    if torch_sec is not None:
        torch_gops = _conv_gops_from_output(expected, in_channels, kernel_height, kernel_width, torch_sec)
        print(f"torch native conv2d: {torch_sec:.4f} sec, {torch_gops:.4f} Gop/s")
        print("torch note: native int16 conv2d returns int16 on this build, so it is a speed baseline only.")
    _print_qpu_benchmark(
        expected,
        in_channels=in_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        prep_sec=prep_sec,
        cached_total_sec=cached_total_sec,
        execute_only_sec=execute_only_sec,
    )
    print(f"Maximum absolute error: {int(np.max(np.abs(diff)))}")

    result = {
        "numpy_sec": time_numpy,
        "qpu_prep_sec": prep_sec,
        "qpu_cached_total_sec": cached_total_sec,
        "qpu_execute_only_sec": execute_only_sec,
        "max_abs_error": float(np.max(np.abs(diff))),
    }
    if torch_sec is not None:
        result["torch_sec"] = torch_sec
    return result


def main() -> None:
    benchmark_tiledconv2d_fp32()
    print()
    benchmark_tiledconv2d_int32()
    print()
    benchmark_tiledconv2d_int16()


if __name__ == "__main__":
    main()
