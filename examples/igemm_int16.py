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
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np
import numpy.typing as npt

from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def gops(p: int, q: int, r: int, sec: float) -> float:
    return (2 * p * q * r) / sec * 1e-9


def pack_int16_pairs(a: npt.NDArray[np.int16]) -> npt.NDArray[np.int32]:
    assert a.ndim == 2
    assert a.shape[1] % 2 == 0

    lo = a[:, 0::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    hi = a[:, 1::2].astype(np.uint16, copy=False).astype(np.uint32, copy=False)
    packed = lo | (hi << 16)
    return np.ascontiguousarray(packed.view(np.int32))


@qpu
def qpu_igemm_rnn_int16_packed(asm: Assembly) -> None:
    # rf0 - rf15: regs
    # rf16 - rf31: 16x16 accumulators
    # rf32 - rf33: unpack helpers
    #
    # A and B are stored as packed int16 pairs in int32 words.
    # The kernel widens each half-word with mov(...unpack("il"/"ih")) and
    # computes with smul24, so the arithmetic still happens in signed int32.

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

    # a_base = packed_a[i, :] = packed_a_item0_addr + 16 * i * a_stride
    # b_base = packed_b[:, j] = packed_b_item0_addr + 16 * j * 4
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

    def emit_packed_mac_step(a_reg: Register, a_half: str, b_reg: Register, b_half: str, *, first: bool = False) -> None:
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

    with loop as lk:
        bnot(tmuc, 3)
        mov(tmua, reg_a_base)

        bnot(tmuc, 3)
        mov(tmua, reg_b_base, sig=thrsw).add(reg_b_base, reg_b_base, reg_b_stride_x4)

        sub(reg_a_base, reg_a_base, -16)

        emit_packed_mac_step(reg_a[0], "il", reg_b[0], "il", first=True)
        emit_packed_mac_step(reg_a[0], "ih", reg_b[0], "ih")
        emit_packed_mac_step(reg_a[1], "il", reg_b[1], "il")
        emit_packed_mac_step(reg_a[1], "ih", reg_b[1], "ih")
        emit_packed_mac_step(reg_a[2], "il", reg_b[2], "il")
        emit_packed_mac_step(reg_a[2], "ih", reg_b[2], "ih")
        emit_packed_mac_step(reg_a[3], "il", reg_b[3], "il")
        emit_packed_mac_step(reg_a[3], "ih", reg_b[3], "ih")

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


def igemm_rnn_int16_packed(p: int = 1024, q: int = 1024, r: int = 1024) -> dict[str, float]:
    assert p > 0
    assert q > 0
    assert r > 0

    assert p % 16 == 0
    assert q % 8 == 0
    assert r % 16 == 0

    tile_p = p // 16
    tile_r = r // 16
    q_packed = q // 2
    data_area_size = (p * q_packed + q_packed * r + p * r) * np.dtype(np.int32).itemsize + 4096

    with Driver(data_area_size=data_area_size) as drv:
        code = drv.program(qpu_igemm_rnn_int16_packed)

        a_ref = np.empty((p, q), dtype=np.int16)
        b_ref = np.empty((q, r), dtype=np.int16)
        np.random.seed(0)

        # Keep the int32 accumulation in range for the default sizes.
        value_limit = 1 << 10
        a_ref[:] = np.random.randint(-value_limit, value_limit, size=a_ref.shape, dtype=np.int16)
        b_ref[:] = np.random.randint(-value_limit, value_limit, size=b_ref.shape, dtype=np.int16)

        a_packed_ref = pack_int16_pairs(a_ref)
        b_packed_ref = pack_int16_pairs(b_ref.T).T.copy()

        a: Array[np.int32] = drv.alloc(a_packed_ref.shape, dtype=np.int32)
        b: Array[np.int32] = drv.alloc(b_packed_ref.shape, dtype=np.int32)
        c: Array[np.int32] = drv.alloc((p, r), dtype=np.int32)

        a[:] = a_packed_ref
        b[:] = b_packed_ref
        c[:] = 0

        start = getsec()
        expected = a_ref.astype(np.int32).dot(b_ref.astype(np.int32))
        time_ref = getsec() - start

        expected_strict = a_ref.astype(np.int64).dot(b_ref.astype(np.int64))
        assert np.max(np.abs(expected_strict)) <= np.iinfo(np.int32).max
        assert np.array_equal(expected, expected_strict.astype(np.int32))

        unif: Array[np.uint32] = drv.alloc(7, dtype=np.uint32)
        unif[0] = a.strides[0]
        unif[1] = a.addresses().item(0)
        unif[2] = b.strides[0]
        unif[3] = b.addresses().item(0)
        unif[4] = c.strides[0]
        unif[5] = c.addresses().item(0)
        unif[6] = q_packed

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

        print(f"==== packed int16 igemm example ({p}x{q} times {q}x{r}) ====")
        print("Kernel contract: A/B are packed int16 pairs in int32 words; accumulation is signed int32 via smul24.")
        print("This reduces input bandwidth, but it is not a packed int8/int16 dot-product kernel.")
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


def sweep_igemm_rnn_int16_packed(sizes: list[int] | tuple[int, ...] = (256, 512, 768, 1024)) -> None:
    results: list[dict[str, float]] = []

    for size in sizes:
        print()
        results.append(igemm_rnn_int16_packed(p=size, q=size, r=size))

    print()
    print("==== packed int16 igemm size sweep summary ====")
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
    sweep_igemm_rnn_int16_packed()


if __name__ == "__main__":
    main()
