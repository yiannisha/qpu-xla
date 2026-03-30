from collections import deque
from time import CLOCK_MONOTONIC, clock_gettime

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from sgemm import qpu_sgemm_rnn_naive
except ImportError:
    try:
        from examples.sgemm import qpu_sgemm_rnn_naive
    except ImportError:
        qpu_sgemm_rnn_naive = None

from videocore7.assembler import *
from videocore7.assembler import Assembly, Register, qpu
from videocore7.driver import Array, Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


VC7_QPUS = 12


@qpu
def qpu_sgemm_rnn_reuse_a_x2(asm: Assembly) -> None:
    # rf16 - rf47: two independent 16x16 accumulator tiles.

    reg_tile_i = rf1
    reg_tile_j_pair = rf2
    reg_a = [rf3, rf4, rf5, rf6]
    reg_b0 = [rf7, rf8, rf9, rf10]
    reg_i = rf11
    reg_a_stride = rf9  # released after address setup
    reg_a_base = rf12
    reg_b_stride = reg_b_stride_x4 = rf13
    reg_b0_base = rf14
    reg_c_stride = rf10  # released after address setup
    reg_c0_base = rf15

    reg_accum0 = [rf[i] for i in range(16, 32)]
    reg_accum1 = [rf[i] for i in range(32, 48)]

    reg_b1 = [rf48, rf49, rf50, rf51]
    reg_b1_base = rf52
    reg_c1_base = rf53
    reg_tmuc_vec_4_cfg = rf54
    reg_alpha = rf55
    reg_beta = rf56
    reg_tmp = rf57
    reg_a3_next = rf58
    reg_mul = rf59

    mov(reg_tile_i, rf3.unpack("uh"))
    mov(reg_tile_j_pair, rf3.unpack("ul"))

    # A base = a[16 * tile_i, :], B0/B1 bases = b[:, 32 * tile_j_pair + {0, 16}]
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

    # Start loading A, B0, and B1.
    bnot(tmuc, 3)
    mov(tmua, reg_a_base)
    bnot(tmuc, 3)
    mov(tmua, reg_b0_base).add(reg_b0_base, reg_b0_base, reg_b_stride_x4)
    bnot(tmuc, 3)
    mov(tmua, reg_b1_base, sig=thrsw).add(reg_b1_base, reg_b1_base, reg_b_stride_x4)

    sub(reg_a_base, reg_a_base, -16)

    # C0/C1 bases = c[16 * tile_i, 32 * tile_j_pair + {0, 16}]
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
        # Start loading the next A, B0, and B1 blocks.
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
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, states[0][2])

        for idx in range(1, len(states) - 1):
            _, prev_accum, _ = states[idx - 1]
            cur_tile, _, cur_a = states[idx]

            rotate_broadcast_reg_b(cur_tile)
            sig = load_sig(idx - 1)
            if sig is None:
                fadd(prev_accum, prev_accum, rf1).fmul(rf1, rf0, cur_a)
            else:
                fadd(prev_accum, prev_accum, rf1, sig=sig).fmul(rf1, rf0, cur_a)

        lk.b(cond="anyna")
        rotate_broadcast_reg_b(states[-1][0])

        sig = load_sig(len(states) - 2)
        if sig is None:
            fadd(states[-2][1], states[-2][1], rf1).fmul(rf1, rf0, states[-1][2])
        else:
            fadd(states[-2][1], states[-2][1], rf1, sig=sig).fmul(rf1, rf0, states[-1][2])

        sig = load_sig(len(states) - 1)
        if sig is None:
            fadd(states[-1][1], states[-1][1], rf1).mov(reg_a[3], reg_a3_next)
        else:
            fadd(states[-1][1], states[-1][1], rf1, sig=sig).mov(reg_a[3], reg_a3_next)

    del reg_a
    del reg_b0
    del reg_b1
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b0_base
    del reg_b1_base

    def store_tile(reg_accum: list[Register], reg_c_base: Register) -> None:
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[0], reg_accum[0], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[1], reg_accum[1], reg_alpha)
        fmul(reg_accum[2], reg_accum[2], reg_alpha)
        fmul(reg_accum[3], reg_accum[3], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[0], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[1], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[2], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[3], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[4], reg_accum[4], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[5], reg_accum[5], reg_alpha)
        fmul(reg_accum[6], reg_accum[6], reg_alpha)
        fmul(reg_accum[7], reg_accum[7], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[4], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[5], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[6], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[7], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[8], reg_accum[8], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[9], reg_accum[9], reg_alpha)
        fmul(reg_accum[10], reg_accum[10], reg_alpha)
        fmul(reg_accum[11], reg_accum[11], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[8], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[9], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[10], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[11], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[12], reg_accum[12], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[13], reg_accum[13], reg_alpha)
        fmul(reg_accum[14], reg_accum[14], reg_alpha)
        fmul(reg_accum[15], reg_accum[15], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[12], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[13], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[14], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[15], rf2)
        mov(tmua, reg_c_base)
        tmuwt()

    bnot(reg_tmuc_vec_4_cfg, 3)
    nop(sig=ldunifrf(reg_alpha))
    nop(sig=ldunifrf(reg_beta))
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
def qpu_sgemm_rnn_reuse_a_x2_qpu_aware(asm: Assembly) -> None:
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
    reg_tmuc_vec_4_cfg = rf54
    reg_alpha = rf55
    reg_beta = rf56
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
        sub(reg_i, reg_i, 1, cond="pushz").fmul(rf1, rf0, states[0][2])

        for idx in range(1, len(states) - 1):
            _, prev_accum, _ = states[idx - 1]
            cur_tile, _, cur_a = states[idx]

            rotate_broadcast_reg_b(cur_tile)
            sig = load_sig(idx - 1)
            if sig is None:
                fadd(prev_accum, prev_accum, rf1).fmul(rf1, rf0, cur_a)
            else:
                fadd(prev_accum, prev_accum, rf1, sig=sig).fmul(rf1, rf0, cur_a)

        lk.b(cond="anyna")
        rotate_broadcast_reg_b(states[-1][0])

        sig = load_sig(len(states) - 2)
        if sig is None:
            fadd(states[-2][1], states[-2][1], rf1).fmul(rf1, rf0, states[-1][2])
        else:
            fadd(states[-2][1], states[-2][1], rf1, sig=sig).fmul(rf1, rf0, states[-1][2])

        sig = load_sig(len(states) - 1)
        if sig is None:
            fadd(states[-1][1], states[-1][1], rf1).mov(reg_a[3], reg_a3_next)
        else:
            fadd(states[-1][1], states[-1][1], rf1, sig=sig).mov(reg_a[3], reg_a3_next)

    del reg_a
    del reg_b0
    del reg_b1
    del reg_i
    del reg_a_base
    del reg_b_stride_x4
    del reg_b0_base
    del reg_b1_base

    def store_tile(reg_accum: list[Register], reg_c_base: Register) -> None:
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[0], reg_accum[0], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[1], reg_accum[1], reg_alpha)
        fmul(reg_accum[2], reg_accum[2], reg_alpha)
        fmul(reg_accum[3], reg_accum[3], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[0], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[1], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[2], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[3], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[4], reg_accum[4], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[5], reg_accum[5], reg_alpha)
        fmul(reg_accum[6], reg_accum[6], reg_alpha)
        fmul(reg_accum[7], reg_accum[7], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[4], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[5], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[6], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[7], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[8], reg_accum[8], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[9], reg_accum[9], reg_alpha)
        fmul(reg_accum[10], reg_accum[10], reg_alpha)
        fmul(reg_accum[11], reg_accum[11], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[8], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[9], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[10], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[11], rf2)
        mov(tmua, reg_c_base).sub(reg_c_base, reg_c_base, -16)

        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(reg_accum[12], reg_accum[12], reg_alpha)
        mov(tmua, reg_c_base, sig=thrsw)
        fmul(reg_accum[13], reg_accum[13], reg_alpha)
        fmul(reg_accum[14], reg_accum[14], reg_alpha)
        fmul(reg_accum[15], reg_accum[15], reg_alpha, sig=ldtmu(rf1))
        mov(tmuc, reg_tmuc_vec_4_cfg).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[12], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[13], rf2).fmul(rf2, rf1, reg_beta, sig=ldtmu(rf1))
        fadd(tmud, reg_accum[14], rf2).fmul(rf2, rf1, reg_beta)
        fadd(tmud, reg_accum[15], rf2)
        mov(tmua, reg_c_base)
        tmuwt()

    bnot(reg_tmuc_vec_4_cfg, 3)
    nop(sig=ldunifrf(reg_alpha))
    nop(sig=ldunifrf(reg_beta))
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


def run_kernel(
    drv: Driver,
    code: Array[np.uint64],
    a: Array[np.float32],
    b: Array[np.float32],
    c: Array[np.float32],
    *,
    q: int,
    alpha: float,
    beta: float,
    workgroup: tuple[int, int, int],
    thread: int,
) -> float:
    unif: Array[np.uint32] = drv.alloc(9, dtype=np.uint32)
    unif[0] = a.strides[0]
    unif[1] = a.addresses().item(0)
    unif[2] = b.strides[0]
    unif[3] = b.addresses().item(0)
    unif[4] = c.strides[0]
    unif[5] = c.addresses().item(0)
    unif[6] = q
    unif[7] = np.float32(alpha).view(np.uint32).item()
    unif[8] = np.float32(beta).view(np.uint32).item()

    start = getsec()
    drv.execute(
        code,
        local_invocation=(16, 1, 1),
        uniforms=unif.addresses().item(0),
        workgroup=workgroup,
        wgs_per_sg=24,
        thread=thread,
    )
    return getsec() - start


def run_qpu_aware_kernel(
    drv: Driver,
    code: Array[np.uint64],
    a: Array[np.float32],
    b: Array[np.float32],
    c: Array[np.float32],
    *,
    q: int,
    alpha: float,
    beta: float,
    tile_p: int,
    tile_r_pair: int,
) -> float:
    total_tasks = tile_p * tile_r_pair
    if total_tasks < VC7_QPUS:
        return run_kernel(
            drv,
            code,
            a,
            b,
            c,
            q=q,
            alpha=alpha,
            beta=beta,
            workgroup=(tile_r_pair, tile_p, 1),
            thread=total_tasks,
        )

    task_counts = [0 for _ in range(VC7_QPUS)]
    for task_idx in range(total_tasks):
        task_counts[task_idx % VC7_QPUS] += 1

    max_tasks = max(task_counts)
    streams: Array[np.uint32] = drv.alloc((VC7_QPUS, max_tasks, 10), dtype=np.uint32)
    meta: Array[np.uint32] = drv.alloc((VC7_QPUS, 16), dtype=np.uint32)
    global_unif: Array[np.uint32] = drv.alloc(2, dtype=np.uint32)

    streams[:] = 0
    meta[:] = 0

    alpha_bits = np.float32(alpha).view(np.uint32).item()
    beta_bits = np.float32(beta).view(np.uint32).item()

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
        meta[qpu_id, 0] = streams.addresses()[qpu_id, 0, 0]
        meta[qpu_id, 1] = count

    global_unif[0] = meta.addresses()[0, 0]
    global_unif[1] = meta.strides[0]

    start = getsec()
    drv.execute(
        code,
        local_invocation=(16, 1, 1),
        uniforms=global_unif.addresses()[0],
        wgs_per_sg=VC7_QPUS,
        thread=VC7_QPUS,
    )
    return getsec() - start


def gflops(p: int, q: int, r: int, sec: float) -> float:
    return (2 * p * q * r + 3 * p * r) / sec * 1e-9


def summarize_error(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    diff = actual - expected
    nan_count = int(np.isnan(actual).sum())
    inf_count = int(np.isinf(actual).sum())
    print(f"{name} NaN count: {nan_count}")
    print(f"{name} Inf count: {inf_count}")
    print(f"{name} minimum absolute error: {np.nanmin(np.abs(diff))}")
    print(f"{name} maximum absolute error: {np.nanmax(np.abs(diff))}")

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rel = np.abs(diff / expected)
    print(f"{name} minimum relative error: {np.nanmin(rel)}")
    print(f"{name} maximum relative error: {np.nanmax(rel)}")


def sgemm_rnn_reuse_a_x2() -> None:
    p = 1024
    q = 1024
    r = 1024

    assert p % 16 == 0
    assert q % 4 == 0
    assert r % 32 == 0

    tile_p = p // 16
    tile_r = r // 16
    tile_r_pair = r // 32

    with Driver() as drv:
        fast_code = drv.program(qpu_sgemm_rnn_reuse_a_x2)
        qpu_aware_code = drv.program(qpu_sgemm_rnn_reuse_a_x2_qpu_aware)
        naive_code = drv.program(qpu_sgemm_rnn_naive) if qpu_sgemm_rnn_naive is not None else None

        a: Array[np.float32] = drv.alloc((p, q), dtype=np.float32)
        b: Array[np.float32] = drv.alloc((q, r), dtype=np.float32)
        c_fast: Array[np.float32] = drv.alloc((p, r), dtype=np.float32)
        c_qpu_aware: Array[np.float32] = drv.alloc((p, r), dtype=np.float32)
        c_naive: Array[np.float32] | None
        if naive_code is None:
            c_naive = None
        else:
            c_naive = drv.alloc((p, r), dtype=np.float32)

        np.random.seed(0)
        alpha = np.random.randn()
        beta = np.random.randn()
        a_ref = np.random.randn(*a.shape).astype(a.dtype)
        b_ref = np.random.randn(*b.shape).astype(b.dtype)
        c_ref = np.random.randn(*c_fast.shape).astype(c_fast.dtype)
        expected = np.empty(c_fast.shape, dtype=c_fast.dtype)

        a[:] = a_ref
        b[:] = b_ref
        c_fast[:] = c_ref
        c_qpu_aware[:] = c_ref
        if c_naive is not None:
            c_naive[:] = c_ref

        start = getsec()
        expected[:] = alpha * a_ref.dot(b_ref) + beta * c_ref
        time_ref = getsec() - start

        time_torch: float | None = None
        if torch is not None:
            a_torch = torch.from_numpy(a_ref)
            b_torch = torch.from_numpy(b_ref)
            c_torch = torch.from_numpy(c_ref)
            start = getsec()
            with torch.no_grad():
                _ = alpha * torch.matmul(a_torch, b_torch) + beta * c_torch
            time_torch = getsec() - start

        time_naive: float | None = None
        if naive_code is not None and c_naive is not None:
            time_naive = run_kernel(
                drv,
                naive_code,
                a,
                b,
                c_naive,
                q=q,
                alpha=alpha,
                beta=beta,
                workgroup=(tile_r, tile_p, 1),
                thread=tile_p * tile_r,
            )

        time_fast = run_kernel(
            drv,
            fast_code,
            a,
            b,
            c_fast,
            q=q,
            alpha=alpha,
            beta=beta,
            workgroup=(tile_r_pair, tile_p, 1),
            thread=tile_p * tile_r_pair,
        )
        time_qpu_aware = run_qpu_aware_kernel(
            drv,
            qpu_aware_code,
            a,
            b,
            c_qpu_aware,
            q=q,
            alpha=alpha,
            beta=beta,
            tile_p=tile_p,
            tile_r_pair=tile_r_pair,
        )

        print(f"==== fast sgemm example ({p}x{q} times {q}x{r}) ====")
        print(f"numpy: {time_ref:.4f} sec, {gflops(p, q, r, time_ref):.4f} Gflop/s")
        if time_torch is None:
            print("torch: n/a (torch is not installed)")
        else:
            print(f"torch: {time_torch:.4f} sec, {gflops(p, q, r, time_torch):.4f} Gflop/s")
        if time_naive is None:
            print("QPU naive: n/a (examples/sgemm.py is not importable)")
        else:
            print(f"QPU naive: {time_naive:.4f} sec, {gflops(p, q, r, time_naive):.4f} Gflop/s")
        print(f"QPU fast payload:    {time_fast:.4f} sec, {gflops(p, q, r, time_fast):.4f} Gflop/s")
        print(f"QPU fast qpu-aware:  {time_qpu_aware:.4f} sec, {gflops(p, q, r, time_qpu_aware):.4f} Gflop/s")
        if time_naive is not None:
            print(f"Speedup over naive: {time_naive / time_fast:.3f}x")
        if c_naive is not None:
            summarize_error("Naive", c_naive, expected)
        summarize_error("Fast payload", c_fast, expected)
        summarize_error("Fast qpu-aware", c_qpu_aware, expected)


def main() -> None:
    sgemm_rnn_reuse_a_x2()


if __name__ == "__main__":
    main()
