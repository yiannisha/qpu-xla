# Copyright (c) 2026- Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
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
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def pool_gib_per_sec(output_elements: int, itemsize: int, sec: float) -> float:
    bytes_moved = output_elements * itemsize * 5
    return bytes_moved / sec / (1024**3)


def benchmark(
    name: str,
    output_elements: int,
    itemsize: int,
    repeat: int,
    fn: Callable[[], np.ndarray],
) -> tuple[np.ndarray, float]:
    result = fn()
    best_sec = float("inf")
    for _ in range(repeat):
        start = getsec()
        result = fn()
        best_sec = min(best_sec, getsec() - start)
    print(f"{name:<10} {best_sec:.6f} sec, {pool_gib_per_sec(output_elements, itemsize, best_sec):.3f} GiB/s")
    return result, best_sec


def pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expected a pair")
        return value
    return (value, value)


def output_hw(height: int, width: int, kernel: tuple[int, int], stride: tuple[int, int]) -> tuple[int, int]:
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    if height % stride_h != 0 or width % stride_w != 0:
        raise ValueError("input dimensions must be divisible by the stride")
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("invalid pool geometry")
    return out_h, out_w


def trunc_divide_pow2_numpy(x: np.ndarray, shift: int) -> np.ndarray:
    x64 = x.astype(np.int64, copy=False)
    bias = ((x64 < 0).astype(np.int64)) * ((1 << shift) - 1)
    return (x64 + bias) >> shift


def pack_int16_adjacent_pairs(x: npt.NDArray[np.int16]) -> npt.NDArray[np.uint32]:
    if x.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {x.shape}")
    if x.shape[-1] % 2 != 0:
        raise ValueError("int16 packed input width must be even")

    lo = x[..., 0::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    hi = x[..., 1::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    return np.ascontiguousarray(lo | (hi << 16))


def unpack_int16_pairs(words: npt.NDArray[np.uint32], *, out_width: int) -> npt.NDArray[np.int16]:
    if words.ndim != 4:
        raise ValueError(f"expected packed output tensor, got shape {words.shape}")
    flat = np.ascontiguousarray(words.reshape(-1))
    out = np.empty(flat.size * 2, dtype=np.int16)
    out[0::2] = (flat & 0xFFFF).astype(np.uint16, copy=False).view(np.int16)
    out[1::2] = ((flat >> 16) & 0xFFFF).astype(np.uint16, copy=False).view(np.int16)
    batch, channels, out_h, packed_w = words.shape
    return out.reshape(batch, channels, out_h, packed_w * 2)[..., :out_width]


def numpy_pool2d_nchw(x: np.ndarray, *, op: str) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected NCHW input, got shape {x.shape}")
    if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
        raise ValueError("this example implements only 2x2 stride-2 pooling on even H/W")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    if op == "max":
        return np.maximum(np.maximum(x00, x01), np.maximum(x10, x11))

    if op != "avg":
        raise ValueError("op must be 'max' or 'avg'")

    if np.issubdtype(x.dtype, np.floating):
        return ((x00 + x01) + (x10 + x11)) * np.float32(0.25)

    acc = x00.astype(np.int64) + x01.astype(np.int64) + x10.astype(np.int64) + x11.astype(np.int64)
    return trunc_divide_pow2_numpy(acc, 2).astype(x.dtype, copy=False)


def torch_pool2d_nchw(x: np.ndarray, *, op: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is not available")

    with torch.no_grad():
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        x00 = x_t[:, :, 0::2, 0::2]
        x01 = x_t[:, :, 0::2, 1::2]
        x10 = x_t[:, :, 1::2, 0::2]
        x11 = x_t[:, :, 1::2, 1::2]

        if op == "max":
            y_t = torch.maximum(torch.maximum(x00, x01), torch.maximum(x10, x11))
        elif op == "avg":
            if torch.is_floating_point(x_t):
                y_t = ((x00 + x01) + (x10 + x11)) * 0.25
            else:
                acc = x00.to(torch.int64) + x01.to(torch.int64) + x10.to(torch.int64) + x11.to(torch.int64)
                y_t = torch.div(acc, 4, rounding_mode="trunc").to(x_t.dtype)
        else:
            raise ValueError("op must be 'max' or 'avg'")

    return y_t.cpu().numpy()


def emit_trunc_div4_int(dst: Register, src: Register, sign: Register, bias: Register, shift31: Register) -> None:
    asr(sign, src, shift31)
    band(bias, sign, 3)
    add(dst, src, bias)
    asr(dst, dst, 2)


@qpu
def qpu_pool2d_fp32(asm: Assembly, *, mode: str, num_qpus: int) -> None:
    if mode not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_row_stride = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_word_stride = rf6
    reg_shift31 = rf7
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
    nop(sig=ldunifrf(reg_row_stride))
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
        nop()
        add(rf31, reg_base, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        add(rf31, reg_base, reg_row_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v1))
        add(rf31, rf31, 4)
        mov(tmua, rf31, sig=thrsw)
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
def qpu_pool2d_int32(asm: Assembly, *, mode: str, num_qpus: int) -> None:
    if mode not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_row_stride = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_word_stride = rf6
    reg_shift31 = rf7
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
    nop(sig=ldunifrf(reg_row_stride))
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
        nop()
        add(rf31, reg_base, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v0))
        add(rf31, reg_base, reg_row_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_v1))
        add(rf31, rf31, 4)
        mov(tmua, rf31, sig=thrsw)
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


@qpu
def qpu_pool2d_int16_packed(asm: Assembly, *, mode: str, num_qpus: int) -> None:
    if mode not in ("max", "avg"):
        raise ValueError("mode must be 'max' or 'avg'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_meta = rf1
    reg_dst = rf2
    reg_row_stride = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_word_stride = rf6
    reg_shift31 = rf7
    reg_base = rf10
    reg_top0 = rf11
    reg_top1 = rf12
    reg_bot0 = rf13
    reg_bot1 = rf14
    reg_tmp = rf15
    reg_out0 = rf16
    reg_out1 = rf17
    reg_sign = rf18
    reg_bias = rf19
    reg_word = rf20

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_meta))
    nop(sig=ldunifrf(reg_dst))
    nop(sig=ldunifrf(reg_row_stride))
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
        nop()
        add(rf31, reg_base, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_top0))
        add(rf31, reg_base, reg_row_stride)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_top1))
        add(rf31, rf31, 4)
        mov(tmua, rf31, sig=thrsw)
        nop(sig=ldtmu(reg_bot0))
        nop()
        nop(sig=ldtmu(reg_bot1))

        if mode == "max":
            mov(reg_tmp, reg_top0.unpack("il"))
            mov(reg_out0, reg_top0.unpack("ih"))
            imax(reg_out0, reg_tmp, reg_out0)
            mov(reg_tmp, reg_bot0.unpack("il"))
            imax(reg_out0, reg_out0, reg_tmp)
            mov(reg_tmp, reg_bot0.unpack("ih"))
            imax(reg_out0, reg_out0, reg_tmp)

            mov(reg_tmp, reg_top1.unpack("il"))
            mov(reg_out1, reg_top1.unpack("ih"))
            imax(reg_out1, reg_tmp, reg_out1)
            mov(reg_tmp, reg_bot1.unpack("il"))
            imax(reg_out1, reg_out1, reg_tmp)
            mov(reg_tmp, reg_bot1.unpack("ih"))
            imax(reg_out1, reg_out1, reg_tmp)
        else:
            mov(reg_out0, reg_top0.unpack("il"))
            mov(reg_tmp, reg_top0.unpack("ih"))
            add(reg_out0, reg_out0, reg_tmp)
            mov(reg_tmp, reg_bot0.unpack("il"))
            add(reg_out0, reg_out0, reg_tmp)
            mov(reg_tmp, reg_bot0.unpack("ih"))
            add(reg_out0, reg_out0, reg_tmp)
            emit_trunc_div4_int(reg_out0, reg_out0, reg_sign, reg_bias, reg_shift31)

            mov(reg_out1, reg_top1.unpack("il"))
            mov(reg_tmp, reg_top1.unpack("ih"))
            add(reg_out1, reg_out1, reg_tmp)
            mov(reg_tmp, reg_bot1.unpack("il"))
            add(reg_out1, reg_out1, reg_tmp)
            mov(reg_tmp, reg_bot1.unpack("ih"))
            add(reg_out1, reg_out1, reg_tmp)
            emit_trunc_div4_int(reg_out1, reg_out1, reg_sign, reg_bias, reg_shift31)

        vpack(reg_word, reg_out0, reg_out1)
        mov(tmud, reg_word)
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


@dataclass(frozen=True)
class DTypeConfig:
    name: str
    np_dtype: np.dtype[Any]
    qpu_kernel: Callable[..., None]


DTYPE_CONFIGS: dict[str, DTypeConfig] = {
    "fp32": DTypeConfig("fp32", np.dtype(np.float32), qpu_pool2d_fp32),
    "int32": DTypeConfig("int32", np.dtype(np.int32), qpu_pool2d_int32),
    "int16": DTypeConfig("int16", np.dtype(np.int16), qpu_pool2d_int16_packed),
}


def validate_geometry(
    *,
    batch: int,
    channels: int,
    height: int,
    width: int,
    num_qpus: int,
    configs: list[DTypeConfig],
) -> dict[str, int]:
    if batch <= 0 or channels <= 0:
        raise ValueError("batch and channels must be positive")
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("this example requires even height and width for 2x2 stride-2 pooling")

    out_h, out_w = output_hw(height, width, (2, 2), (2, 2))
    iterations: dict[str, int] = {}

    for config in configs:
        if config.name == "int16":
            if out_w % 2 != 0:
                raise ValueError("int16 kernel packs two adjacent outputs, so output width must be even")
            output_words = batch * channels * out_h * (out_w // 2)
            chunk = 16 * num_qpus
            if output_words % chunk != 0:
                raise ValueError(f"int16 output words must be a multiple of {chunk}")
            iterations[config.name] = output_words // chunk
        else:
            output_scalars = batch * channels * out_h * out_w
            chunk = 16 * num_qpus
            if output_scalars % chunk != 0:
                raise ValueError(f"{config.name} output elements must be a multiple of {chunk}")
            iterations[config.name] = output_scalars // chunk

    return iterations


def required_data_area_size(config: DTypeConfig, *, batch: int, channels: int, height: int, width: int) -> int:
    out_h, out_w = output_hw(height, width, (2, 2), (2, 2))

    if config.name == "int16":
        input_bytes = batch * channels * height * (width // 2) * 4
        output_words = batch * channels * out_h * (out_w // 2)
        output_bytes = output_words * 4
        meta_bytes = output_words * 4
        return input_bytes + output_bytes + meta_bytes + 4096

    input_bytes = batch * channels * height * width * config.np_dtype.itemsize
    output_bytes = batch * channels * out_h * out_w * config.np_dtype.itemsize
    meta_bytes = batch * channels * out_h * out_w * 4
    return input_bytes + output_bytes + meta_bytes + 4096


def random_input(config: DTypeConfig, *, batch: int, channels: int, height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (batch, channels, height, width)
    if config.name == "fp32":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if config.name == "int32":
        return rng.integers(-(2**27), 2**27, size=shape, dtype=np.int32)
    if config.name == "int16":
        return rng.integers(-(2**14), 2**14, size=shape, dtype=np.int16)
    raise RuntimeError("unreachable")


def make_scalar_meta(x_dev: Array[Any]) -> np.ndarray:
    return np.ascontiguousarray(x_dev.addresses()[:, :, 0::2, 0::2].reshape(-1).astype(np.uint32, copy=False))


def make_packed_int16_meta(x_packed_dev: Array[np.uint32]) -> np.ndarray:
    return np.ascontiguousarray(x_packed_dev.addresses()[:, :, 0::2, 0::2].reshape(-1).astype(np.uint32, copy=False))


def run_qpu_pool2d(
    *,
    config: DTypeConfig,
    x: np.ndarray,
    mode: str,
    iterations: int,
    num_qpus: int,
    repeat: int,
) -> tuple[np.ndarray, float]:
    batch, channels, height, width = x.shape
    out_h, out_w = output_hw(height, width, (2, 2), (2, 2))

    data_area_size = required_data_area_size(
        config,
        batch=batch,
        channels=channels,
        height=height,
        width=width,
    )
    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(config.qpu_kernel, mode=mode, num_qpus=num_qpus)

        if config.name == "int16":
            x_packed_host = pack_int16_adjacent_pairs(x.astype(np.int16, copy=False))
            x_dev: Array[np.uint32] = drv.alloc(x_packed_host.shape, dtype=np.uint32)
            x_dev[:] = x_packed_host

            meta_host = make_packed_int16_meta(x_dev)
            meta_dev: Array[np.uint32] = drv.alloc(meta_host.shape, dtype=np.uint32)
            meta_dev[:] = meta_host

            out_packed_shape = (batch, channels, out_h, out_w // 2)
            out_dev: Array[np.uint32] = drv.alloc(out_packed_shape, dtype=np.uint32)
            out_dev[:] = 0

            uniforms: Array[np.uint32] = drv.alloc(4, dtype=np.uint32)
            uniforms[0] = iterations
            uniforms[1] = meta_dev.addresses()[0]
            uniforms[2] = out_dev.addresses()[0, 0, 0, 0]
            uniforms[3] = x_dev.strides[2]

            def run() -> np.ndarray:
                drv.execute(
                    code,
                    local_invocation=(16, 1, 1),
                    uniforms=uniforms.addresses()[0],
                    thread=num_qpus,
                )
                return unpack_int16_pairs(out_dev, out_width=out_w)

            return benchmark("QPU:", batch * channels * out_h * out_w, config.np_dtype.itemsize, repeat, run)

        x_dev = drv.alloc(x.shape, dtype=config.np_dtype)
        x_dev[:] = x

        meta_host = make_scalar_meta(x_dev)
        meta_dev = drv.alloc(meta_host.shape, dtype=np.uint32)
        meta_dev[:] = meta_host

        out_shape = (batch, channels, out_h, out_w)
        out_dev = drv.alloc(out_shape, dtype=config.np_dtype)
        out_dev[:] = 0

        uniforms = drv.alloc(4, dtype=np.uint32)
        uniforms[0] = iterations
        uniforms[1] = meta_dev.addresses()[0]
        uniforms[2] = out_dev.addresses()[0, 0, 0, 0]
        uniforms[3] = x_dev.strides[2]

        def run() -> np.ndarray:
            drv.execute(
                code,
                local_invocation=(16, 1, 1),
                uniforms=uniforms.addresses()[0],
                thread=num_qpus,
            )
            return np.array(out_dev, copy=True)

        return benchmark("QPU:", batch * channels * out_h * out_w, config.np_dtype.itemsize, repeat, run)


def assert_close(config: DTypeConfig, actual: np.ndarray, expected: np.ndarray) -> None:
    if config.name == "fp32":
        if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            diff = np.abs(actual - expected)
            raise AssertionError(
                f"{config.name} mismatch: max_abs_diff={float(np.max(diff))}, first_bad_index={int(np.argmax(diff))}"
            )
        return

    if not np.array_equal(actual, expected):
        diff = actual.astype(np.int64) - expected.astype(np.int64)
        flat = np.flatnonzero(diff)
        raise AssertionError(
            f"{config.name} mismatch: max_abs_diff={int(np.max(np.abs(diff)))}, first_bad_index={int(flat[0])}"
        )


def run_dtype_benchmarks(
    *,
    config: DTypeConfig,
    batch: int,
    channels: int,
    height: int,
    width: int,
    iterations: int,
    num_qpus: int,
    repeat: int,
    seed: int,
) -> None:
    x = random_input(config, batch=batch, channels=channels, height=height, width=width, seed=seed)
    out_h, out_w = output_hw(height, width, (2, 2), (2, 2))
    qpu_label = f"{num_qpus} QPU{'s'[: num_qpus - 1]}"
    title = (
        f"==== {config.name} 2x2/2 pool "
        f"({batch}x{channels}x{height}x{width} -> {batch}x{channels}x{out_h}x{out_w}, {qpu_label}) ===="
    )
    print(title)

    for mode in ("max", "avg"):
        print(f"{mode}pool:")

        expected, _ = benchmark(
            "numpy:",
            batch * channels * out_h * out_w,
            config.np_dtype.itemsize,
            repeat,
            lambda: numpy_pool2d_nchw(x, op=mode),
        )

        if torch is None:
            print("torch:      n/a (torch is not installed)")
        else:
            torch_out, _ = benchmark(
                "torch:",
                batch * channels * out_h * out_w,
                config.np_dtype.itemsize,
                repeat,
                lambda: torch_pool2d_nchw(x, op=mode),
            )
            assert_close(config, torch_out, expected)

        qpu_out, _ = run_qpu_pool2d(
            config=config,
            x=x,
            mode=mode,
            iterations=iterations,
            num_qpus=num_qpus,
            repeat=repeat,
        )
        assert_close(config, qpu_out, expected)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 2x2 stride-2 maxpool/avgpool QPU kernels against NumPy and Torch"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-qpus", type=int, default=12, choices=(1, 12))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPE_CONFIGS),
        default=["fp32", "int32", "int16"],
        help="Dtypes to benchmark",
    )
    args = parser.parse_args()

    configs = [DTYPE_CONFIGS[name] for name in args.dtypes]
    iterations_by_dtype = validate_geometry(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        num_qpus=args.num_qpus,
        configs=configs,
    )

    print("Pool contract: NCHW, kernel=2x2, stride=2, padding=0.")
    print("Integer avgpool uses truncation toward zero to match the Torch baseline.")
    print(f"dtypes: {', '.join(args.dtypes)}")
    print()

    for idx, config in enumerate(configs):
        run_dtype_benchmarks(
            config=config,
            batch=args.batch,
            channels=args.channels,
            height=args.height,
            width=args.width,
            iterations=iterations_by_dtype[config.name],
            num_qpus=args.num_qpus,
            repeat=args.repeat,
            seed=args.seed,
        )
        if idx != len(configs) - 1:
            print()


if __name__ == "__main__":
    main()
