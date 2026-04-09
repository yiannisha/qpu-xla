from __future__ import annotations

import importlib.util
import sys
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

from videocore7.driver import Array, Driver


def _load_tiledmlp_module() -> Any:
    module_path = Path(__file__).with_name("tiledmlp.py")
    spec = importlib.util.spec_from_file_location("_tiledattention_tiledmlp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TILEDMLP = _load_tiledmlp_module()

P_TILE = int(_TILEDMLP.P_TILE)
Q_TILE = int(_TILEDMLP.Q_TILE)
SIGNED_24BIT_LIMIT = int(_TILEDMLP.SIGNED_24BIT_LIMIT)
VALIDATED_LINEAR_R_TILE = int(_TILEDMLP.VALIDATED_LINEAR_R_TILE)

_execute_linear_dispatch = _TILEDMLP._execute_linear_dispatch
_build_validated_tiled_matmul_dispatch = _TILEDMLP._build_validated_tiled_matmul_dispatch
qpu_tiledconv2d_fp32 = _TILEDMLP.qpu_tiledconv2d_fp32
qpu_tiledconv2d_int32 = _TILEDMLP.qpu_tiledconv2d_int32


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def _round_up(value: int, tile: int) -> int:
    return ((value + tile - 1) // tile) * tile


def attention_gops(
    query_len: int,
    key_len: int,
    depth: int,
    value_dim: int,
    sec: float,
) -> float:
    ops = 2 * query_len * key_len * depth + 2 * query_len * key_len * value_dim
    return ops / sec * 1e-9


def attention_score_gops(
    query_len: int,
    key_len: int,
    depth: int,
    sec: float,
) -> float:
    ops = 2 * query_len * key_len * depth
    return ops / sec * 1e-9


def attention_value_gops(
    query_len: int,
    key_len: int,
    value_dim: int,
    sec: float,
) -> float:
    ops = 2 * query_len * key_len * value_dim
    return ops / sec * 1e-9


def _max_abs_int(x: npt.NDArray[np.integer[Any]]) -> int:
    if x.size == 0:
        return 0
    x64 = x.astype(np.int64, copy=False)
    return int(np.max(np.abs(x64)))


def _ensure_int32_range(x: npt.NDArray[np.int64], *, what: str) -> None:
    max_abs = int(np.max(np.abs(x), initial=0))
    if max_abs > np.iinfo(np.int32).max:
        raise ValueError(f"{what} exceeds the signed int32 range")


def _validate_attention_shapes(
    q: npt.NDArray[np.generic],
    k: npt.NDArray[np.generic],
    v: npt.NDArray[np.generic],
) -> tuple[int, int, int, int]:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("attention expects 2-D matrices for q, k, and v")
    if q.shape[1] != k.shape[1]:
        raise ValueError("query and key depth do not match")
    if k.shape[0] != v.shape[0]:
        raise ValueError("key length and value rows do not match")
    return int(q.shape[0]), int(k.shape[0]), int(q.shape[1]), int(v.shape[1])


def _validate_int32_attention_contract(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> None:
    _, key_len, depth, _ = _validate_attention_shapes(q, k, v)

    max_q = _max_abs_int(q)
    max_k = _max_abs_int(k)
    max_v = _max_abs_int(v)

    if max(max_q, max_k, max_v) >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 attention kernels use smul24; q, k, and v must fit the signed 24-bit range")

    score_bound = depth * max_q * max_k
    if score_bound >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 attention score matrix may exceed the signed 24-bit range required by the value stage")

    output_bound = key_len * score_bound * max_v
    if output_bound > np.iinfo(np.int32).max:
        raise ValueError("int32 attention output may exceed the signed int32 accumulation range")


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


def numpy_attention_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    scores = numpy_attention_scores_fp32(q, k)
    return numpy_attention_value_fp32(scores, v)


def numpy_attention_scores_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("attention score stage expects 2-D matrices")
    if q.shape[1] != k.shape[1]:
        raise ValueError("query and key depth do not match")
    if q.dtype != np.float32 or k.dtype != np.float32:
        raise ValueError("fp32 attention score stage expects float32 inputs")

    q_contig = np.ascontiguousarray(q)
    k_contig = np.ascontiguousarray(k)
    return np.ascontiguousarray(q_contig.dot(k_contig.T), dtype=np.float32)


def numpy_attention_value_fp32(
    scores: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if scores.ndim != 2 or v.ndim != 2:
        raise ValueError("attention value stage expects 2-D matrices")
    if scores.shape[1] != v.shape[0]:
        raise ValueError("score width and value rows do not match")
    if scores.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("fp32 attention value stage expects float32 inputs")

    scores_contig = np.ascontiguousarray(scores)
    v_contig = np.ascontiguousarray(v)
    return np.ascontiguousarray(scores_contig.dot(v_contig), dtype=np.float32)


def numpy_sdpa_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    scores = numpy_attention_scores_fp32(q, k)
    scores_max = np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores - scores_max)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum
    return numpy_attention_value_fp32(weights.astype(np.float32, copy=False), v)


def reference_attention_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    scores = reference_attention_scores_int32(q, k)
    return reference_attention_value_int32(scores, v)


def reference_attention_scores_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("attention score stage expects 2-D matrices")
    if q.shape[1] != k.shape[1]:
        raise ValueError("query and key depth do not match")
    if q.dtype != np.int32 or k.dtype != np.int32:
        raise ValueError("int32 attention score stage expects int32 inputs")
    if max(_max_abs_int(q), _max_abs_int(k)) >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 attention score stage uses smul24; q and k must fit the signed 24-bit range")

    scores = q.astype(np.int64).dot(k.astype(np.int64).T)
    max_scores = int(np.max(np.abs(scores), initial=0))
    if max_scores >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 attention score matrix may exceed the signed 24-bit range required by the value stage")
    return np.ascontiguousarray(scores.astype(np.int32))


def reference_attention_value_int32(
    scores: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    _validate_int32_value_stage_contract(scores, v)

    output = scores.astype(np.int64).dot(v.astype(np.int64))
    _ensure_int32_range(output, what="int32 attention output")
    return np.ascontiguousarray(output.astype(np.int32))


def numpy_attention_scores_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    reference_attention_scores_int32(q, k)
    q_contig = np.ascontiguousarray(q)
    k_contig = np.ascontiguousarray(k)
    return np.ascontiguousarray(q_contig.dot(k_contig.T), dtype=np.int32)


def numpy_attention_value_int32(
    scores: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    _validate_int32_value_stage_contract(scores, v)
    scores_contig = np.ascontiguousarray(scores)
    v_contig = np.ascontiguousarray(v)
    return np.ascontiguousarray(scores_contig.dot(v_contig), dtype=np.int32)


def _validate_int32_value_stage_contract(
    scores: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> None:
    if scores.ndim != 2 or v.ndim != 2:
        raise ValueError("attention value stage expects 2-D matrices")
    if scores.shape[1] != v.shape[0]:
        raise ValueError("score width and value rows do not match")
    if scores.dtype != np.int32 or v.dtype != np.int32:
        raise ValueError("int32 attention value stage expects int32 inputs")

    max_scores = _max_abs_int(scores)
    max_v = _max_abs_int(v)
    if max(max_scores, max_v) >= SIGNED_24BIT_LIMIT:
        raise ValueError("int32 attention value stage uses smul24; scores and v must fit the signed 24-bit range")

    output_bound = scores.shape[1] * max_scores * max_v
    if output_bound > np.iinfo(np.int32).max:
        raise ValueError("int32 attention output may exceed the signed int32 accumulation range")


def numpy_attention_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    _validate_int32_attention_contract(q, k, v)
    scores = numpy_attention_scores_int32(q, k)
    return numpy_attention_value_int32(scores, v)


def torch_attention_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    scores = torch_attention_scores_fp32(q, k)
    return torch_attention_value_fp32(scores, v)


def torch_attention_scores_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if torch is None:
        raise RuntimeError("torch is not available")

    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("attention score stage expects 2-D matrices")
    if q.shape[1] != k.shape[1]:
        raise ValueError("query and key depth do not match")
    if q.dtype != np.float32 or k.dtype != np.float32:
        raise ValueError("fp32 attention score stage expects float32 inputs")

    with torch.no_grad():
        q_t = torch.from_numpy(np.ascontiguousarray(q))
        k_t = torch.from_numpy(np.ascontiguousarray(k))
        out_t = q_t.matmul(k_t.transpose(0, 1))
    return out_t.cpu().numpy()


def torch_attention_value_fp32(
    scores: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if torch is None:
        raise RuntimeError("torch is not available")

    if scores.ndim != 2 or v.ndim != 2:
        raise ValueError("attention value stage expects 2-D matrices")
    if scores.shape[1] != v.shape[0]:
        raise ValueError("score width and value rows do not match")
    if scores.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("fp32 attention value stage expects float32 inputs")

    with torch.no_grad():
        scores_t = torch.from_numpy(np.ascontiguousarray(scores))
        v_t = torch.from_numpy(np.ascontiguousarray(v))
        out_t = scores_t.matmul(v_t)
    return out_t.cpu().numpy()


def torch_attention_sdpa_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if torch is None or torch_f is None or not hasattr(torch_f, "scaled_dot_product_attention"):
        raise RuntimeError("torch scaled_dot_product_attention is not available")

    _validate_attention_shapes(q, k, v)
    if q.dtype != np.float32 or k.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("fp32 attention expects float32 inputs")

    with torch.no_grad():
        q_t = torch.from_numpy(np.ascontiguousarray(q))[None, None, :, :]
        k_t = torch.from_numpy(np.ascontiguousarray(k))[None, None, :, :]
        v_t = torch.from_numpy(np.ascontiguousarray(v))[None, None, :, :]
        out_t = torch_f.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
    return out_t[0, 0].cpu().numpy()


def torch_attention_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    scores = torch_attention_scores_int32(q, k)
    return torch_attention_value_int32(scores, v)


def torch_attention_scores_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    if torch is None:
        raise RuntimeError("torch is not available")

    reference_attention_scores_int32(q, k)

    with torch.no_grad():
        q_t = torch.from_numpy(np.ascontiguousarray(q))
        k_t = torch.from_numpy(np.ascontiguousarray(k))
        out_t = q_t.matmul(k_t.transpose(0, 1))
    return out_t.cpu().numpy()


def torch_attention_value_int32(
    scores: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    if torch is None:
        raise RuntimeError("torch is not available")

    _validate_int32_value_stage_contract(scores, v)

    with torch.no_grad():
        scores_t = torch.from_numpy(np.ascontiguousarray(scores))
        v_t = torch.from_numpy(np.ascontiguousarray(v))
        out_t = scores_t.matmul(v_t)
    return out_t.cpu().numpy()


@dataclass(frozen=True)
class PreparedAttentionProblem:
    q: npt.NDArray[np.generic]
    k_t: npt.NDArray[np.generic]
    v: npt.NDArray[np.generic]
    query_len: int
    key_len: int
    value_dim: int


@dataclass(frozen=True)
class PreparedAttentionValueStage:
    scores: npt.NDArray[np.generic]
    v: npt.NDArray[np.generic]
    query_len: int
    value_dim: int


@dataclass(frozen=True)
class QpuBenchmarkStats:
    prep_sec: float
    cached_total_sec: float
    execute_only_sec: float
    max_abs_error: float


class _BaseTiledAttentionExecutor:
    query_len: int
    key_len: int
    depth: int
    value_dim: int
    query_padded: int
    key_padded: int
    depth_padded: int
    value_padded: int
    _drv: Driver
    _q_dev: Array[np.generic]
    _k_t_dev: Array[np.generic]
    _v_dev: Array[np.generic]
    _scores_dev: Array[np.generic]
    _output_dev: Array[np.generic]
    _score_dispatch: Any
    _output_dispatch: Any

    def close(self) -> None:
        self._drv.close()

    def __enter__(self) -> "_BaseTiledAttentionExecutor":
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
    ) -> npt.NDArray[np.generic]:
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {x.shape}")
        if x.dtype != dtype:
            raise ValueError(f"expected dtype {dtype}, got {x.dtype}")
        if x.shape != expected_shape:
            raise ValueError(f"expected shape {expected_shape}, got {x.shape}")
        matrix = np.zeros(padded_shape, dtype=dtype)
        matrix[: expected_shape[0], : expected_shape[1]] = x
        return np.ascontiguousarray(matrix)

    def prepare_problem(
        self,
        q: npt.NDArray[np.generic],
        k: npt.NDArray[np.generic],
        v: npt.NDArray[np.generic],
    ) -> PreparedAttentionProblem:
        _validate_attention_shapes(q, k, v)
        if q.shape != (self.query_len, self.depth):
            raise ValueError(f"expected q shape {(self.query_len, self.depth)}, got {q.shape}")
        if k.shape != (self.key_len, self.depth):
            raise ValueError(f"expected k shape {(self.key_len, self.depth)}, got {k.shape}")
        if v.shape != (self.key_len, self.value_dim):
            raise ValueError(f"expected v shape {(self.key_len, self.value_dim)}, got {v.shape}")
        if q.dtype != self._q_dev.dtype or k.dtype != self._k_t_dev.dtype or v.dtype != self._v_dev.dtype:
            raise ValueError(f"expected dtype {self._q_dev.dtype} for q, k, and v")

        q_padded = self._prepare_matrix(
            q,
            expected_shape=(self.query_len, self.depth),
            padded_shape=(self.query_padded, self.depth_padded),
            dtype=self._q_dev.dtype,
        )
        k_t_padded = np.zeros((self.depth_padded, self.key_padded), dtype=self._k_t_dev.dtype)
        k_t_padded[: self.depth, : self.key_len] = np.ascontiguousarray(k.T)
        v_padded = self._prepare_matrix(
            v,
            expected_shape=(self.key_len, self.value_dim),
            padded_shape=(self.key_padded, self.value_padded),
            dtype=self._v_dev.dtype,
        )
        return PreparedAttentionProblem(
            q=q_padded,
            k_t=np.ascontiguousarray(k_t_padded),
            v=v_padded,
            query_len=self.query_len,
            key_len=self.key_len,
            value_dim=self.value_dim,
        )

    def prepare_scores(
        self,
        scores: npt.NDArray[np.generic],
    ) -> npt.NDArray[np.generic]:
        return self._prepare_matrix(
            scores,
            expected_shape=(self.query_len, self.key_len),
            padded_shape=(self.query_padded, self.key_padded),
            dtype=self._scores_dev.dtype,
        )

    def prepare_value_stage(
        self,
        scores: npt.NDArray[np.generic],
        v: npt.NDArray[np.generic],
    ) -> PreparedAttentionValueStage:
        scores_padded = self.prepare_scores(scores)
        v_padded = self._prepare_matrix(
            v,
            expected_shape=(self.key_len, self.value_dim),
            padded_shape=(self.key_padded, self.value_padded),
            dtype=self._v_dev.dtype,
        )
        return PreparedAttentionValueStage(
            scores=scores_padded,
            v=v_padded,
            query_len=self.query_len,
            value_dim=self.value_dim,
        )

    def upload(self, prepared: PreparedAttentionProblem) -> None:
        self._q_dev[:] = prepared.q
        self._k_t_dev[:] = prepared.k_t
        self._v_dev[:] = prepared.v

    def upload_scores(
        self,
        scores: npt.NDArray[np.generic],
    ) -> None:
        self._scores_dev[:] = scores

    def upload_value_stage(
        self,
        prepared: PreparedAttentionValueStage,
    ) -> None:
        self._scores_dev[:] = prepared.scores
        self._v_dev[:] = prepared.v

    def execute_scores(self) -> None:
        _execute_linear_dispatch(self._drv, self._score_dispatch)

    def execute_values(self) -> None:
        _execute_linear_dispatch(self._drv, self._output_dispatch)

    def execute_attention(self) -> None:
        self.execute_scores()
        self.execute_values()

    def read_scores(self) -> npt.NDArray[np.generic]:
        scores = np.array(self._scores_dev, copy=True)
        return np.ascontiguousarray(scores[: self.query_len, : self.key_len])

    def read_output(self) -> npt.NDArray[np.generic]:
        output = np.array(self._output_dev, copy=True)
        return np.ascontiguousarray(output[: self.query_len, : self.value_dim])


class TiledAttentionExecutorFP32(_BaseTiledAttentionExecutor):
    def __init__(
        self,
        *,
        query_len: int,
        key_len: int,
        depth: int,
        value_dim: int,
    ) -> None:
        if query_len <= 0 or key_len <= 0 or depth <= 0 or value_dim <= 0:
            raise ValueError("attention dimensions must be positive")

        self.query_len = int(query_len)
        self.key_len = int(key_len)
        self.depth = int(depth)
        self.value_dim = int(value_dim)
        self.query_padded = _round_up(self.query_len, P_TILE)
        self.key_padded = _round_up(self.key_len, VALIDATED_LINEAR_R_TILE)
        self.depth_padded = _round_up(self.depth, Q_TILE)
        self.value_padded = _round_up(self.value_dim, VALIDATED_LINEAR_R_TILE)

        data_area_size = (
            self.query_padded * self.depth_padded
            + self.depth_padded * self.key_padded
            + self.key_padded * self.value_padded
            + self.query_padded * self.key_padded
            + self.query_padded * self.value_padded
        ) * np.dtype(np.float32).itemsize + (1 << 20)

        self._drv = Driver(data_area_size=data_area_size)
        tiled_matmul_code = self._drv.program(qpu_tiledconv2d_fp32)

        self._q_dev = self._drv.alloc((self.query_padded, self.depth_padded), dtype=np.float32)
        self._k_t_dev = self._drv.alloc((self.depth_padded, self.key_padded), dtype=np.float32)
        self._v_dev = self._drv.alloc((self.key_padded, self.value_padded), dtype=np.float32)
        self._scores_dev = self._drv.alloc((self.query_padded, self.key_padded), dtype=np.float32)
        self._output_dev = self._drv.alloc((self.query_padded, self.value_padded), dtype=np.float32)

        self._scores_dev[:] = 0.0
        self._output_dev[:] = 0.0

        self._score_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._q_dev,
            self._k_t_dev,
            self._scores_dev,
            q=self.depth_padded,
        )
        self._output_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._scores_dev,
            self._v_dev,
            self._output_dev,
            q=self.key_padded,
        )


class TiledAttentionExecutorInt32(_BaseTiledAttentionExecutor):
    def __init__(
        self,
        *,
        query_len: int,
        key_len: int,
        depth: int,
        value_dim: int,
    ) -> None:
        if query_len <= 0 or key_len <= 0 or depth <= 0 or value_dim <= 0:
            raise ValueError("attention dimensions must be positive")

        self.query_len = int(query_len)
        self.key_len = int(key_len)
        self.depth = int(depth)
        self.value_dim = int(value_dim)
        self.query_padded = _round_up(self.query_len, P_TILE)
        self.key_padded = _round_up(self.key_len, VALIDATED_LINEAR_R_TILE)
        self.depth_padded = _round_up(self.depth, Q_TILE)
        self.value_padded = _round_up(self.value_dim, VALIDATED_LINEAR_R_TILE)

        data_area_size = (
            self.query_padded * self.depth_padded
            + self.depth_padded * self.key_padded
            + self.key_padded * self.value_padded
            + self.query_padded * self.key_padded
            + self.query_padded * self.value_padded
        ) * np.dtype(np.int32).itemsize + (1 << 20)

        self._drv = Driver(data_area_size=data_area_size)
        tiled_matmul_code = self._drv.program(qpu_tiledconv2d_int32)

        self._q_dev = self._drv.alloc((self.query_padded, self.depth_padded), dtype=np.int32)
        self._k_t_dev = self._drv.alloc((self.depth_padded, self.key_padded), dtype=np.int32)
        self._v_dev = self._drv.alloc((self.key_padded, self.value_padded), dtype=np.int32)
        self._scores_dev = self._drv.alloc((self.query_padded, self.key_padded), dtype=np.int32)
        self._output_dev = self._drv.alloc((self.query_padded, self.value_padded), dtype=np.int32)

        self._scores_dev[:] = 0
        self._output_dev[:] = 0

        self._score_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._q_dev,
            self._k_t_dev,
            self._scores_dev,
            q=self.depth_padded,
        )
        self._output_dispatch = _build_validated_tiled_matmul_dispatch(
            self._drv,
            tiled_matmul_code,
            self._scores_dev,
            self._v_dev,
            self._output_dev,
            q=self.key_padded,
        )

    def prepare_problem(
        self,
        q: npt.NDArray[np.int32],
        k: npt.NDArray[np.int32],
        v: npt.NDArray[np.int32],
    ) -> PreparedAttentionProblem:
        _validate_int32_attention_contract(q, k, v)
        return super().prepare_problem(q, k, v)

    def prepare_scores(
        self,
        scores: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        if scores.ndim != 2:
            raise ValueError("attention score stage expects a 2-D score matrix")
        if scores.shape != (self.query_len, self.key_len):
            raise ValueError(f"expected score shape {(self.query_len, self.key_len)}, got {scores.shape}")
        if scores.dtype != np.int32:
            raise ValueError(f"expected dtype {np.int32}, got {scores.dtype}")
        if _max_abs_int(scores) >= SIGNED_24BIT_LIMIT:
            raise ValueError("int32 attention value stage uses smul24; scores must fit the signed 24-bit range")
        return super().prepare_scores(scores)

    def prepare_value_stage(
        self,
        scores: npt.NDArray[np.int32],
        v: npt.NDArray[np.int32],
    ) -> PreparedAttentionValueStage:
        _validate_int32_value_stage_contract(scores, v)
        return super().prepare_value_stage(scores, v)


def tiledattention_fp32(
    q: npt.NDArray[np.float32],
    k: npt.NDArray[np.float32],
    v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    _validate_attention_shapes(q, k, v)
    if q.dtype != np.float32 or k.dtype != np.float32 or v.dtype != np.float32:
        raise ValueError("fp32 attention expects float32 inputs")

    with TiledAttentionExecutorFP32(
        query_len=int(q.shape[0]),
        key_len=int(k.shape[0]),
        depth=int(q.shape[1]),
        value_dim=int(v.shape[1]),
    ) as executor:
        prepared = executor.prepare_problem(q, k, v)
        executor.upload(prepared)
        executor.execute_attention()
        return executor.read_output()


def tiledattention_int32(
    q: npt.NDArray[np.int32],
    k: npt.NDArray[np.int32],
    v: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    _validate_int32_attention_contract(q, k, v)

    with TiledAttentionExecutorInt32(
        query_len=int(q.shape[0]),
        key_len=int(k.shape[0]),
        depth=int(q.shape[1]),
        value_dim=int(v.shape[1]),
    ) as executor:
        prepared = executor.prepare_problem(q, k, v)
        executor.upload(prepared)
        executor.execute_attention()
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


def _print_attention_stats(
    title: str,
    *,
    numpy_sec: float,
    torch_sec: float | None,
    throughput_fn,
    qpu_stats: QpuBenchmarkStats | None,
    torch_label: str,
    extra_lines: list[str] | None = None,
) -> None:
    print(f"-- {title} --")
    print(f"numpy: {numpy_sec:.4f} sec, {throughput_fn(numpy_sec):.4f} Gop/s")
    if torch_sec is None:
        print(f"{torch_label}: n/a")
    else:
        print(f"{torch_label}: {torch_sec:.4f} sec, {throughput_fn(torch_sec):.4f} Gop/s")
    if qpu_stats is None:
        if extra_lines is not None:
            for line in extra_lines:
                print(line)
        print("QPU: unavailable")
        print()
        return

    if extra_lines is not None:
        for line in extra_lines:
            print(line)
    prep_cached_sec = qpu_stats.prep_sec + qpu_stats.cached_total_sec
    print(f"QPU host prep: {qpu_stats.prep_sec:.4f} sec")
    print(
        f"QPU cached total: {qpu_stats.cached_total_sec:.4f} sec, "
        f"{throughput_fn(qpu_stats.cached_total_sec):.4f} Gop/s"
    )
    print(
        f"QPU execute only: {qpu_stats.execute_only_sec:.4f} sec, "
        f"{throughput_fn(qpu_stats.execute_only_sec):.4f} Gop/s"
    )
    print(f"QPU prep+cached total: {prep_cached_sec:.4f} sec, {throughput_fn(prep_cached_sec):.4f} Gop/s")
    print(f"Maximum absolute error: {qpu_stats.max_abs_error}")
    print()


def benchmark_tiledattention_fp32() -> dict[str, float]:
    query_len = 128
    key_len = 128
    depth = 128
    value_dim = 128

    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=(query_len, depth)).astype(np.float32)
    k = rng.uniform(-1.0, 1.0, size=(key_len, depth)).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, size=(key_len, value_dim)).astype(np.float32)

    expected_scores, numpy_score_sec = _benchmark_callable(lambda: numpy_attention_scores_fp32(q, k), repeat=3)
    expected_value_output, numpy_value_sec = _benchmark_callable(
        lambda: numpy_attention_value_fp32(expected_scores, v), repeat=3
    )
    expected, numpy_total_sec = _benchmark_callable(lambda: numpy_attention_fp32(q, k, v), repeat=3)
    assert np.allclose(expected, expected_value_output, atol=1e-5, rtol=1e-5)

    torch_score_sec = None
    torch_value_sec = None
    torch_total_sec = None
    torch_sdpa_sec = None
    if torch is not None:
        torch_scores, torch_score_sec = _benchmark_callable(lambda: torch_attention_scores_fp32(q, k), repeat=5)
        torch_value_output, torch_value_sec = _benchmark_callable(
            lambda: torch_attention_value_fp32(expected_scores, v),
            repeat=5,
        )
        torch_output, torch_total_sec = _benchmark_callable(lambda: torch_attention_fp32(q, k, v), repeat=5)
        assert np.allclose(torch_scores, expected_scores, atol=1e-4, rtol=1e-4)
        assert np.allclose(torch_value_output, expected_value_output, atol=1e-4, rtol=1e-4)
        assert np.allclose(torch_output, expected, atol=1e-4, rtol=1e-4)

        try:
            expected_sdpa = numpy_sdpa_fp32(q, k, v)
            torch_sdpa_output, torch_sdpa_sec = _benchmark_callable(lambda: torch_attention_sdpa_fp32(q, k, v), repeat=5)
            assert np.allclose(torch_sdpa_output, expected_sdpa, atol=1e-4, rtol=1e-4)
        except RuntimeError:
            torch_sdpa_sec = None

    print("==== tiledattention fp32 example ====")
    print("Operator: single-head dot-product attention core O = (Q @ K^T) @ V")
    print(f"Dimensions: q={query_len}x{depth}, k={key_len}x{depth}, v={key_len}x{value_dim}")
    print("Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.")

    result: dict[str, float] = {"numpy_sec": numpy_total_sec}
    setup_start = getsec()
    try:
        executor = TiledAttentionExecutorFP32(
            query_len=query_len,
            key_len=key_len,
            depth=depth,
            value_dim=value_dim,
        )
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        _print_attention_stats(
            "Score stage (Q @ K^T)",
            numpy_sec=numpy_score_sec,
            torch_sec=torch_score_sec,
            throughput_fn=lambda sec: attention_score_gops(query_len, key_len, depth, sec),
            qpu_stats=None,
            torch_label="torch matmul",
        )
        _print_attention_stats(
            "Value stage (Scores @ V, reference scores)",
            numpy_sec=numpy_value_sec,
            torch_sec=torch_value_sec,
            throughput_fn=lambda sec: attention_value_gops(query_len, key_len, value_dim, sec),
            qpu_stats=None,
            torch_label="torch matmul",
        )
        total_extra_lines = []
        if torch_sdpa_sec is None:
            total_extra_lines.append("torch native sdpa: n/a")
        else:
            total_extra_lines.append(
                f"torch native sdpa: {torch_sdpa_sec:.4f} sec, "
                f"{attention_gops(query_len, key_len, depth, value_dim, torch_sdpa_sec):.4f} Gop/s"
            )
        total_extra_lines.append(
            "torch native sdpa note: includes softmax with scale=1.0; speed baseline only, not a correctness reference."
        )
        _print_attention_stats(
            "Attention total",
            numpy_sec=numpy_total_sec,
            torch_sec=torch_total_sec,
            throughput_fn=lambda sec: attention_gops(query_len, key_len, depth, value_dim, sec),
            qpu_stats=None,
            torch_label="torch matmul attention core",
            extra_lines=total_extra_lines,
        )
        if torch_total_sec is not None:
            result["torch_sec"] = torch_total_sec
        if torch_sdpa_sec is not None:
            result["torch_sdpa_sec"] = torch_sdpa_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        score_actual, score_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_problem(q, k, v),
            executor.upload,
            executor.execute_scores,
            executor.read_scores,
        )
        score_stats = _with_error(score_stats, actual=score_actual, expected=expected_scores)

        value_actual, value_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_value_stage(expected_scores, v),
            executor.upload_value_stage,
            executor.execute_values,
            executor.read_output,
        )
        value_stats = _with_error(value_stats, actual=value_actual, expected=expected_value_output)

        actual, qpu_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_problem(q, k, v),
            executor.upload,
            executor.execute_attention,
            executor.read_output,
        )
        qpu_stats = _with_error(qpu_stats, actual=actual, expected=expected)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    _print_attention_stats(
        "Score stage (Q @ K^T)",
        numpy_sec=numpy_score_sec,
        torch_sec=torch_score_sec,
        throughput_fn=lambda sec: attention_score_gops(query_len, key_len, depth, sec),
        qpu_stats=score_stats,
        torch_label="torch matmul",
    )
    _print_attention_stats(
        "Value stage (Scores @ V, reference scores)",
        numpy_sec=numpy_value_sec,
        torch_sec=torch_value_sec,
        throughput_fn=lambda sec: attention_value_gops(query_len, key_len, value_dim, sec),
        qpu_stats=value_stats,
        torch_label="torch matmul",
    )
    total_extra_lines = []
    if torch_sdpa_sec is None:
        total_extra_lines.append("torch native sdpa: n/a")
    else:
        total_extra_lines.append(
            f"torch native sdpa: {torch_sdpa_sec:.4f} sec, "
            f"{attention_gops(query_len, key_len, depth, value_dim, torch_sdpa_sec):.4f} Gop/s"
        )
    total_extra_lines.append(
        "torch native sdpa note: includes softmax with scale=1.0; speed baseline only, not a correctness reference."
    )
    _print_attention_stats(
        "Attention total",
        numpy_sec=numpy_total_sec,
        torch_sec=torch_total_sec,
        throughput_fn=lambda sec: attention_gops(query_len, key_len, depth, value_dim, sec),
        qpu_stats=qpu_stats,
        torch_label="torch matmul attention core",
        extra_lines=total_extra_lines,
    )

    result.update(
        {
            "torch_sec": -1.0 if torch_total_sec is None else torch_total_sec,
            "torch_sdpa_sec": -1.0 if torch_sdpa_sec is None else torch_sdpa_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": qpu_stats.prep_sec,
            "qpu_cached_total_sec": qpu_stats.cached_total_sec,
            "qpu_execute_only_sec": qpu_stats.execute_only_sec,
            "score_stage_max_abs_error": score_stats.max_abs_error,
            "value_stage_max_abs_error": value_stats.max_abs_error,
            "max_abs_error": qpu_stats.max_abs_error,
        }
    )
    return result


def benchmark_tiledattention_int32() -> dict[str, float]:
    query_len = 128
    key_len = 128
    depth = 128
    value_dim = 128

    rng = np.random.default_rng(1)
    q = rng.integers(-8, 8, size=(query_len, depth), dtype=np.int32)
    k = rng.integers(-8, 8, size=(key_len, depth), dtype=np.int32)
    v = rng.integers(-8, 8, size=(key_len, value_dim), dtype=np.int32)

    _validate_int32_attention_contract(q, k, v)
    expected_scores = reference_attention_scores_int32(q, k)
    expected_value_output = reference_attention_value_int32(expected_scores, v)
    expected = reference_attention_int32(q, k, v)
    numpy_scores_output, numpy_score_sec = _benchmark_callable(lambda: numpy_attention_scores_int32(q, k), repeat=3)
    numpy_value_output, numpy_value_sec = _benchmark_callable(lambda: numpy_attention_value_int32(expected_scores, v), repeat=3)
    numpy_output, numpy_total_sec = _benchmark_callable(lambda: numpy_attention_int32(q, k, v), repeat=3)
    np.testing.assert_array_equal(numpy_scores_output, expected_scores)
    np.testing.assert_array_equal(numpy_value_output, expected_value_output)
    np.testing.assert_array_equal(numpy_output, expected)

    torch_score_sec = None
    torch_value_sec = None
    torch_total_sec = None
    torch_error: str | None = None
    if torch is not None:
        try:
            torch_scores, torch_score_sec = _benchmark_callable(lambda: torch_attention_scores_int32(q, k), repeat=5)
            torch_value_output, torch_value_sec = _benchmark_callable(
                lambda: torch_attention_value_int32(expected_scores, v),
                repeat=5,
            )
            torch_output, torch_total_sec = _benchmark_callable(lambda: torch_attention_int32(q, k, v), repeat=5)
            np.testing.assert_array_equal(torch_scores, expected_scores)
            np.testing.assert_array_equal(torch_value_output, expected_value_output)
            np.testing.assert_array_equal(torch_output, expected)
        except RuntimeError as exc:
            torch_error = str(exc)

    print("==== tiledattention int32 example ====")
    print("Operator: single-head dot-product attention core O = (Q @ K^T) @ V")
    print(f"Dimensions: q={query_len}x{depth}, k={key_len}x{depth}, v={key_len}x{value_dim}")
    print("Kernel contract: q, k, and v must fit the signed 24-bit range, and the intermediate score matrix must also stay within it.")
    print("Benchmark mode: steady-state QPU timings use precompiled kernels and persistent device buffers.")
    if torch_error is not None:
        print(f"Torch int32 path unavailable: {torch_error}")

    result: dict[str, float] = {"numpy_sec": numpy_total_sec}
    setup_start = getsec()
    try:
        executor = TiledAttentionExecutorInt32(
            query_len=query_len,
            key_len=key_len,
            depth=depth,
            value_dim=value_dim,
        )
    except Exception as exc:
        print(f"QPU benchmark unavailable: {exc}")
        _print_attention_stats(
            "Score stage (Q @ K^T)",
            numpy_sec=numpy_score_sec,
            torch_sec=torch_score_sec,
            throughput_fn=lambda sec: attention_score_gops(query_len, key_len, depth, sec),
            qpu_stats=None,
            torch_label="torch int32 matmul",
        )
        _print_attention_stats(
            "Value stage (Scores @ V, reference scores)",
            numpy_sec=numpy_value_sec,
            torch_sec=torch_value_sec,
            throughput_fn=lambda sec: attention_value_gops(query_len, key_len, value_dim, sec),
            qpu_stats=None,
            torch_label="torch int32 matmul",
        )
        _print_attention_stats(
            "Attention total",
            numpy_sec=numpy_total_sec,
            torch_sec=torch_total_sec,
            throughput_fn=lambda sec: attention_gops(query_len, key_len, depth, value_dim, sec),
            qpu_stats=None,
            torch_label="torch int32 attention core",
        )
        if torch_total_sec is not None:
            result["torch_sec"] = torch_total_sec
        result["qpu_available"] = 0.0
        return result

    with executor:
        setup_sec = getsec() - setup_start
        score_actual, score_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_problem(q, k, v),
            executor.upload,
            executor.execute_scores,
            executor.read_scores,
        )
        score_stats = _with_error(score_stats, actual=score_actual, expected=expected_scores)

        value_actual, value_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_value_stage(expected_scores, v),
            executor.upload_value_stage,
            executor.execute_values,
            executor.read_output,
        )
        value_stats = _with_error(value_stats, actual=value_actual, expected=expected_value_output)

        actual, qpu_stats = _benchmark_qpu_operator(
            lambda: executor.prepare_problem(q, k, v),
            executor.upload,
            executor.execute_attention,
            executor.read_output,
        )
        qpu_stats = _with_error(qpu_stats, actual=actual, expected=expected)

    print(f"QPU setup (excluded): {setup_sec:.4f} sec")
    _print_attention_stats(
        "Score stage (Q @ K^T)",
        numpy_sec=numpy_score_sec,
        torch_sec=torch_score_sec,
        throughput_fn=lambda sec: attention_score_gops(query_len, key_len, depth, sec),
        qpu_stats=score_stats,
        torch_label="torch int32 matmul",
    )
    _print_attention_stats(
        "Value stage (Scores @ V, reference scores)",
        numpy_sec=numpy_value_sec,
        torch_sec=torch_value_sec,
        throughput_fn=lambda sec: attention_value_gops(query_len, key_len, value_dim, sec),
        qpu_stats=value_stats,
        torch_label="torch int32 matmul",
    )
    _print_attention_stats(
        "Attention total",
        numpy_sec=numpy_total_sec,
        torch_sec=torch_total_sec,
        throughput_fn=lambda sec: attention_gops(query_len, key_len, depth, value_dim, sec),
        qpu_stats=qpu_stats,
        torch_label="torch int32 attention core",
    )

    result.update(
        {
            "torch_sec": -1.0 if torch_total_sec is None else torch_total_sec,
            "qpu_available": 1.0,
            "qpu_setup_sec": setup_sec,
            "qpu_prep_sec": qpu_stats.prep_sec,
            "qpu_cached_total_sec": qpu_stats.cached_total_sec,
            "qpu_execute_only_sec": qpu_stats.execute_only_sec,
            "score_stage_max_abs_error": score_stats.max_abs_error,
            "value_stage_max_abs_error": value_stats.max_abs_error,
            "max_abs_error": qpu_stats.max_abs_error,
        }
    )
    return result


def main() -> None:
    benchmark_tiledattention_fp32()
    print()
    benchmark_tiledattention_int32()


if __name__ == "__main__":
    main()
