# Copyright (c) 2025- Idein Inc.
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
from collections import deque
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def gops(p: int, q: int, r: int, sec: float) -> float:
    return (2 * p * q * r) / sec * 1e-9


@qpu
def qpu_igemm_rnn_naive(asm: Assembly) -> None:
    # rf0 - rf15: regs
    # rf16 - rf31: 16x16 accumulators
    # rf32 - rf63: (unused)
    #
    # This kernel uses smul24, so inputs must fit the signed 24-bit range.

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

    # a_base = a[i, :] = a_item0_addr + 16 * i * a_stride
    # b_base = b[:, j] = b_item0_addr + 16 * j * 4
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

        # 16x4x16 signed integer MAC.
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


def igemm_rnn_naive(p: int = 1024, q: int = 1024, r: int = 1024) -> dict[str, float]:
    assert p > 0
    assert q > 0
    assert r > 0

    assert p % 16 == 0
    assert q % 4 == 0
    assert r % 16 == 0

    tile_p = p // 16
    tile_r = r // 16
    data_area_size = (p * q + q * r + p * r) * np.dtype(np.int32).itemsize + 4096

    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(qpu_igemm_rnn_naive)

        a: Array[np.int32] = drv.alloc((p, q), dtype=np.int32)
        b: Array[np.int32] = drv.alloc((q, r), dtype=np.int32)
        c: Array[np.int32] = drv.alloc((p, r), dtype=np.int32)

        np.random.seed(0)
        value_limit = 1 << 7
        a_ref = np.random.randint(-value_limit, value_limit, size=a.shape, dtype=a.dtype)
        b_ref = np.random.randint(-value_limit, value_limit, size=b.shape, dtype=b.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c[:] = 0

        start = getsec()
        expected = a_ref.dot(b_ref)
        time_ref = getsec() - start

        expected_strict = a_ref.astype(np.int64).dot(b_ref.astype(np.int64)).astype(np.int32)
        assert np.array_equal(expected, expected_strict)

        unif: Array[np.uint32] = drv.alloc(7, dtype=np.uint32)
        unif[0] = a.strides[0]
        unif[1] = a.addresses().item(0)
        unif[2] = b.strides[0]
        unif[3] = b.addresses().item(0)
        unif[4] = c.strides[0]
        unif[5] = c.addresses().item(0)
        unif[6] = q

        start = getsec()
        drv.execute(
            code,
            local_invocation=(16, 1, 1),
            uniforms=unif.addresses().item(0),
            workgroup=(tile_r, tile_p, 1),
            wgs_per_sg=24,
            thread=tile_p * tile_r,
        )
        time_gpu = getsec() - start

        diff = c.astype(np.int64) - expected.astype(np.int64)
        max_abs_error = int(np.max(np.abs(diff)))

        print(f"==== igemm example ({p}x{q} times {q}x{r}) ====")
        print("Kernel contract: inputs must fit in signed 24-bit integers because it uses smul24.")
        print(f"numpy: {time_ref:.4f} sec, {gops(p, q, r, time_ref):.4f} Gop/s")
        print(f"QPU:   {time_gpu:.4f} sec, {gops(p, q, r, time_gpu):.4f} Gop/s")
        print(f"Maximum absolute error: {max_abs_error}")

        return {
            "p": float(p),
            "q": float(q),
            "r": float(r),
            "numpy_sec": time_ref,
            "numpy_gops": gops(p, q, r, time_ref),
            "qpu_sec": time_gpu,
            "qpu_gops": gops(p, q, r, time_gpu),
            "max_abs_error": float(max_abs_error),
        }


def sweep_igemm_rnn_naive(sizes: list[int] | tuple[int, ...] = (256, 512, 768, 1024)) -> None:
    results: list[dict[str, float]] = []

    for size in sizes:
        print()
        results.append(igemm_rnn_naive(p=size, q=size, r=size))

    print()
    print("==== igemm size sweep summary ====")
    print(f"{'size':>6} {'numpy GO/s':>12} {'QPU GO/s':>12} {'QPU sec':>10} {'max abs err':>14}")
    for result in results:
        size = int(result["p"])
        print(
            f"{size:>6} "
            f"{result['numpy_gops']:>12.4f} "
            f"{result['qpu_gops']:>12.4f} "
            f"{result['qpu_sec']:>10.4f} "
            f"{int(result['max_abs_error']):>14}"
        )


def main() -> None:
    sweep_igemm_rnn_naive()


if __name__ == "__main__":
    main()
