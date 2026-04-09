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

try:
    import torch
except ImportError:
    torch = None

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def median_sec(times: list[float]) -> float:
    return float(np.median(np.asarray(times, dtype=np.float64)))


def gib_per_sec(length: int, itemsize: int, sec: float) -> float:
    bytes_moved = length * itemsize * 3
    return bytes_moved / sec / (1024**3)


def pack_int16_pairs(a: np.ndarray) -> np.ndarray:
    a_i16 = np.ascontiguousarray(a.astype(np.int16, copy=False))
    if a_i16.ndim != 1 or a_i16.size % 2 != 0:
        raise ValueError("int16 input must be a flat array with an even length")
    pairs = a_i16.reshape(-1, 2)
    lo = pairs[:, 0].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    hi = pairs[:, 1].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    return (lo | (hi << 16)).astype(np.uint32, copy=False)


def unpack_int16_pairs(words: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(words.astype(np.uint32, copy=False)).reshape(-1)
    out = np.empty(packed.size * 2, dtype=np.int16)
    out[0::2] = (packed & 0xFFFF).astype(np.uint16, copy=False).view(np.int16)
    out[1::2] = ((packed >> 16) & 0xFFFF).astype(np.uint16, copy=False).view(np.int16)
    return out


@qpu
def qpu_binary_float_minmax(asm: Assembly, *, op: str, num_qpus: int) -> None:
    if op not in ("fmin", "fmax"):
        raise ValueError("op must be 'fmin' or 'fmax'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_x = rf1
    reg_y = rf2
    reg_dst = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_x_val = rf10
    reg_y_val = rf11
    reg_out = rf12

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_y))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_stride, 3)
        shl(reg_stride, reg_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_x, reg_x, reg_offset)
    add(reg_y, reg_y, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    op_fn = globals()[op]

    with loop as l:  # noqa: E741
        mov(tmua, reg_x, sig=thrsw).add(reg_x, reg_x, reg_stride)
        nop()
        mov(tmua, reg_y, sig=thrsw).add(reg_y, reg_y, reg_stride)
        nop(sig=ldtmu(reg_x_val))
        nop()
        nop(sig=ldtmu(reg_y_val))

        op_fn(reg_out, reg_x_val, reg_y_val)
        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)
        tmuwt()

        l.b(cond="na0")
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
def qpu_binary_int32_minmax(asm: Assembly, *, op: str, num_qpus: int) -> None:
    if op not in ("imin", "imax"):
        raise ValueError("op must be 'imin' or 'imax'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_x = rf1
    reg_y = rf2
    reg_dst = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_x_val = rf10
    reg_y_val = rf11
    reg_out = rf12

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_y))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_stride, 3)
        shl(reg_stride, reg_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_x, reg_x, reg_offset)
    add(reg_y, reg_y, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    op_fn = globals()[op]

    with loop as l:  # noqa: E741
        mov(tmua, reg_x, sig=thrsw).add(reg_x, reg_x, reg_stride)
        nop()
        mov(tmua, reg_y, sig=thrsw).add(reg_y, reg_y, reg_stride)
        nop(sig=ldtmu(reg_x_val))
        nop()
        nop(sig=ldtmu(reg_y_val))

        op_fn(reg_out, reg_x_val, reg_y_val)
        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)
        tmuwt()

        l.b(cond="na0")
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
def qpu_binary_int16_minmax_packed(asm: Assembly, *, op: str, num_qpus: int) -> None:
    if op not in ("imin", "imax"):
        raise ValueError("op must be 'imin' or 'imax'")
    if num_qpus not in (1, 12):
        raise ValueError("num_qpus must be 1 or 12")

    reg_iters = rf0
    reg_x = rf1
    reg_y = rf2
    reg_dst = rf3
    reg_qpu_num = rf4
    reg_offset = rf5
    reg_stride = rf6
    reg_x_word = rf10
    reg_y_word = rf11
    reg_x_lo = rf12
    reg_y_lo = rf13
    reg_lo = rf14
    reg_x_hi = rf15
    reg_y_hi = rf16
    reg_hi = rf17
    reg_out = rf18

    nop(sig=ldunifrf(reg_iters))
    nop(sig=ldunifrf(reg_x))
    nop(sig=ldunifrf(reg_y))
    nop(sig=ldunifrf(reg_dst))

    if num_qpus == 1:
        mov(reg_qpu_num, 0)
        mov(reg_stride, 1)
        shl(reg_stride, reg_stride, 6)
    else:
        tidx(reg_offset)
        shr(reg_offset, reg_offset, 2)
        band(reg_qpu_num, reg_offset, 0b1111)
        mov(reg_stride, 3)
        shl(reg_stride, reg_stride, 8)

    shl(reg_offset, reg_qpu_num, 4)
    eidx(rf31)
    add(reg_offset, reg_offset, rf31)
    shl(reg_offset, reg_offset, 2)
    add(reg_x, reg_x, reg_offset)
    add(reg_y, reg_y, reg_offset)
    add(reg_dst, reg_dst, reg_offset)

    op_fn = globals()[op]

    with loop as l:  # noqa: E741
        mov(tmua, reg_x, sig=thrsw).add(reg_x, reg_x, reg_stride)
        nop()
        mov(tmua, reg_y, sig=thrsw).add(reg_y, reg_y, reg_stride)
        nop(sig=ldtmu(reg_x_word))
        nop()
        nop(sig=ldtmu(reg_y_word))

        mov(reg_x_lo, reg_x_word.unpack("il"))
        mov(reg_y_lo, reg_y_word.unpack("il"))
        op_fn(reg_lo, reg_x_lo, reg_y_lo)

        mov(reg_x_hi, reg_x_word.unpack("ih"))
        mov(reg_y_hi, reg_y_word.unpack("ih"))
        op_fn(reg_hi, reg_x_hi, reg_y_hi)

        vpack(reg_out, reg_lo, reg_hi)
        mov(tmud, reg_out)
        sub(reg_iters, reg_iters, 1, cond="pushz")
        mov(tmua, reg_dst).add(reg_dst, reg_dst, reg_stride)
        tmuwt()

        l.b(cond="na0")
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


@dataclass(frozen=True)
class DTypeConfig:
    name: str
    np_dtype: np.dtype[Any]
    qpu_kernel: Callable[..., None]
    qpu_ops: dict[str, str]
    elements_per_lane_word: int


DTYPE_CONFIGS: dict[str, DTypeConfig] = {
    "fp32": DTypeConfig(
        name="fp32",
        np_dtype=np.dtype(np.float32),
        qpu_kernel=qpu_binary_float_minmax,
        qpu_ops={"min": "fmin", "max": "fmax"},
        elements_per_lane_word=1,
    ),
    "int32": DTypeConfig(
        name="int32",
        np_dtype=np.dtype(np.int32),
        qpu_kernel=qpu_binary_int32_minmax,
        qpu_ops={"min": "imin", "max": "imax"},
        elements_per_lane_word=1,
    ),
    "int16": DTypeConfig(
        name="int16",
        np_dtype=np.dtype(np.int16),
        qpu_kernel=qpu_binary_int16_minmax_packed,
        qpu_ops={"min": "imin", "max": "imax"},
        elements_per_lane_word=2,
    ),
}


def required_data_area_size(config: DTypeConfig, length: int) -> int:
    if config.name == "int16":
        packed_words = length // 2
        return packed_words * np.dtype(np.uint32).itemsize * 3 + 4096
    return length * config.np_dtype.itemsize * 3 + 4096


def validate_length(length: int, num_qpus: int, configs: list[DTypeConfig]) -> dict[str, int]:
    if length <= 0:
        raise ValueError("length must be positive")

    iterations: dict[str, int] = {}
    for config in configs:
        chunk = 16 * num_qpus * config.elements_per_lane_word
        if length % chunk != 0:
            raise ValueError(f"length must be a multiple of {chunk} for dtype={config.name} and num_qpus={num_qpus}")
        iterations[config.name] = length // chunk
    return iterations


def benchmark(name: str, length: int, itemsize: int, repeat: int, fn: Callable[[], None]) -> float:
    fn()
    times: list[float] = []
    for _ in range(repeat):
        start = getsec()
        fn()
        times.append(getsec() - start)
    sec = median_sec(times)
    print(f"{name:<10} {sec:.6f} sec, {gib_per_sec(length=length, itemsize=itemsize, sec=sec):.3f} GiB/s")
    return sec


def benchmark_numpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, *, op: str, repeat: int) -> float:
    fn = np.minimum if op == "min" else np.maximum
    return benchmark("numpy:", x.size, x.dtype.itemsize, repeat, lambda: fn(x, y, out=out))


def benchmark_torch(x: np.ndarray, y: np.ndarray, *, op: str, repeat: int) -> tuple[float | None, np.ndarray | None]:
    if torch is None:
        print("torch:      n/a (torch is not installed)")
        return None, None

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    out_t = torch.empty_like(x_t)
    fn = torch.minimum if op == "min" else torch.maximum

    def run() -> None:
        with torch.no_grad():
            fn(x_t, y_t, out=out_t)

    sec = benchmark("torch:", x.size, x.dtype.itemsize, repeat, run)
    return sec, out_t.numpy()


def benchmark_qpu(
    drv: Driver,
    code: Array[np.uint64],
    uniforms_addr: int,
    *,
    length: int,
    itemsize: int,
    num_qpus: int,
    repeat: int,
) -> float:
    def run() -> None:
        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=uniforms_addr,
            thread=num_qpus,
        )

    return benchmark("QPU:", length, itemsize, repeat, run)


def run_qpu_operation(
    *,
    config: DTypeConfig,
    drv: Driver,
    x: np.ndarray,
    y: np.ndarray,
    expected: np.ndarray,
    iterations: int,
    num_qpus: int,
    repeat: int,
    op: str,
) -> None:
    code = drv.program(config.qpu_kernel, op=config.qpu_ops[op], num_qpus=num_qpus)

    if config.name == "int16":
        x_qpu: Array[np.uint32] = drv.alloc(x.size // 2, dtype=np.uint32)
        y_qpu: Array[np.uint32] = drv.alloc(y.size // 2, dtype=np.uint32)
        out_qpu: Array[np.uint32] = drv.alloc(expected.size // 2, dtype=np.uint32)

        x_qpu[:] = pack_int16_pairs(x)
        y_qpu[:] = pack_int16_pairs(y)
        out_qpu[:] = 0

        uniforms: Array[np.uint32] = drv.alloc(4, dtype=np.uint32)
        uniforms[0] = iterations
        uniforms[1] = x_qpu.addresses()[0]
        uniforms[2] = y_qpu.addresses()[0]
        uniforms[3] = out_qpu.addresses()[0]

        benchmark_qpu(
            drv,
            code,
            uniforms.addresses()[0],
            length=expected.size,
            itemsize=config.np_dtype.itemsize,
            num_qpus=num_qpus,
            repeat=repeat,
        )

        actual = unpack_int16_pairs(out_qpu)
    else:
        x_qpu: Array[Any] = drv.alloc(x.size, dtype=config.np_dtype)
        y_qpu: Array[Any] = drv.alloc(y.size, dtype=config.np_dtype)
        out_qpu: Array[Any] = drv.alloc(expected.size, dtype=config.np_dtype)

        x_qpu[:] = x
        y_qpu[:] = y
        out_qpu[:] = 0

        uniforms = drv.alloc(4, dtype=np.uint32)
        uniforms[0] = iterations
        uniforms[1] = x_qpu.addresses()[0]
        uniforms[2] = y_qpu.addresses()[0]
        uniforms[3] = out_qpu.addresses()[0]

        benchmark_qpu(
            drv,
            code,
            uniforms.addresses()[0],
            length=expected.size,
            itemsize=config.np_dtype.itemsize,
            num_qpus=num_qpus,
            repeat=repeat,
        )

        actual = np.asarray(out_qpu)

    if not np.array_equal(actual, expected):
        if np.issubdtype(expected.dtype, np.floating):
            diff = np.abs(actual - expected)
            max_abs_diff = float(np.max(diff))
            first_bad_index = int(np.argmax(diff))
        else:
            bad = np.flatnonzero(actual != expected)
            first_bad_index = int(bad[0])
            max_abs_diff = int(np.max(np.abs(actual.astype(np.int64) - expected.astype(np.int64))))
        raise AssertionError(
            f"QPU {config.name} {op} mismatch: max_abs_diff={max_abs_diff}, first_bad_index={first_bad_index}"
        )


def run_dtype_benchmarks(
    *,
    config: DTypeConfig,
    length: int,
    iterations: int,
    num_qpus: int,
    repeat: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if config.name == "fp32":
        x = rng.uniform(-1024.0, 1024.0, size=length).astype(np.float32)
        y = rng.uniform(-1024.0, 1024.0, size=length).astype(np.float32)
    elif config.name == "int32":
        x = rng.integers(-(2**30), 2**30, size=length, dtype=np.int32)
        y = rng.integers(-(2**30), 2**30, size=length, dtype=np.int32)
    elif config.name == "int16":
        x = rng.integers(-(2**14), 2**14, size=length, dtype=np.int16)
        y = rng.integers(-(2**14), 2**14, size=length, dtype=np.int16)
    else:
        raise RuntimeError("unreachable")

    expected_min = np.minimum(x, y)
    expected_max = np.maximum(x, y)

    qpu_label = f"{num_qpus} QPU{'s'[: num_qpus - 1]}"
    size_mi = length / 1024 / 1024
    print(f"==== {config.name} elementwise min/max ({size_mi:.1f} Mi elements, {qpu_label}) ====")

    numpy_out = np.empty_like(x)
    benchmark_numpy(x, y, numpy_out, op="min", repeat=repeat)
    if not np.array_equal(numpy_out, expected_min):
        raise AssertionError(f"NumPy {config.name} minimum result mismatch")
    _, torch_min = benchmark_torch(x, y, op="min", repeat=repeat)
    if torch_min is not None and not np.array_equal(torch_min, expected_min):
        raise AssertionError(f"Torch {config.name} minimum result mismatch")

    with Driver(data_area_size=required_data_area_size(config, length)) as drv:
        run_qpu_operation(
            config=config,
            drv=drv,
            x=x,
            y=y,
            expected=expected_min,
            iterations=iterations,
            num_qpus=num_qpus,
            repeat=repeat,
            op="min",
        )

    print()

    benchmark_numpy(x, y, numpy_out, op="max", repeat=repeat)
    if not np.array_equal(numpy_out, expected_max):
        raise AssertionError(f"NumPy {config.name} maximum result mismatch")
    _, torch_max = benchmark_torch(x, y, op="max", repeat=repeat)
    if torch_max is not None and not np.array_equal(torch_max, expected_max):
        raise AssertionError(f"Torch {config.name} maximum result mismatch")

    with Driver(data_area_size=required_data_area_size(config, length)) as drv:
        run_qpu_operation(
            config=config,
            drv=drv,
            x=x,
            y=y,
            expected=expected_max,
            iterations=iterations,
            num_qpus=num_qpus,
            repeat=repeat,
            op="max",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QPU min/max kernels against NumPy and Torch")
    parser.add_argument("--length", type=int, default=4 * 1024 * 1024, help="Number of scalar elements")
    parser.add_argument("--num-qpus", type=int, default=12, choices=(1, 12), help="Number of QPUs to use")
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repetitions after warmup")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPE_CONFIGS),
        default=["fp32", "int32", "int16"],
        help="Dtypes to benchmark",
    )
    args = parser.parse_args()

    configs = [DTYPE_CONFIGS[name] for name in args.dtypes]
    iterations_by_dtype = validate_length(args.length, args.num_qpus, configs)

    print(f"length: {args.length} scalar values")
    print(f"repeat: {args.repeat}")
    print(f"dtypes: {', '.join(args.dtypes)}")
    print()

    for idx, config in enumerate(configs):
        run_dtype_benchmarks(
            config=config,
            length=args.length,
            iterations=iterations_by_dtype[config.name],
            num_qpus=args.num_qpus,
            repeat=args.repeat,
            seed=args.seed,
        )
        if idx != len(configs) - 1:
            print()


if __name__ == "__main__":
    main()
