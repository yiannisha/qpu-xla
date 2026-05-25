# qpu-xla: QPU-first ML kernels for Raspberry Pi 5

`qpu-xla` is a research runtime stack for executing small ML operators on the
Raspberry Pi 5 VideoCore VII QPU. It builds on
[`py-videocore7`](https://github.com/Idein/py-videocore7) for assembly,
loading, and dispatch, then adds custom tiled kernels for integer GEMM,
GEMM-backed convolution, pooling, min/max, an attention-style core, persistent
executors, and an end-to-end LeNet-style pipeline.

This repository accompanies the MLSys 2026 Young Professionals Symposium paper
"Toward a Small ML Runtime Stack for Raspberry Pi 5 QPUs".

## Hardware Requirements

- Raspberry Pi 5 with VideoCore VII QPU.
- Linux with the DRM V3D device exposed at `/dev/dri/card0`.
- User access to the `video` group, or root:

```console
sudo usermod --append --groups video "$USER"
```

Log out and back in after changing group membership.

## Installation

Install `uv`, clone the repository, and let `uv` resolve the direct
`py-videocore7` dependency from GitHub:

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/yiannisha/qpu-xla.git
cd qpu-xla
uv sync
```

PyTorch is optional and is only used for CPU baseline comparisons in selected
scripts:

```console
uv sync --extra baselines
```

## Tutorial

Run scripts directly with `uv run`. All QPU scripts must be run on Raspberry Pi
5 hardware with V3D access.

```console
uv run examples/igemm.py
uv run examples/igemm_int16.py
uv run examples/sgemm_fast.py
```

Operator-level kernels:

```console
uv run examples/minmax.py
uv run examples/pool2d.py
uv run examples/tiledconv2d.py
uv run examples/tiledattention.py
```

End-to-end LeNet-style pipeline:

```console
uv run examples/tiledlenet5.py
```

The scripts report NumPy, optional PyTorch, cached QPU, and execute-only QPU
measurements where applicable. Cached QPU timing excludes one-time setup such
as assembly, buffer allocation, and metadata construction; execute-only timing
isolates kernel dispatch and device execution.

## Paper Metrics

Metrics below are transcribed from `MLSYS26_YPS.pdf`.

### Integer GEMM Throughput

| Size | INT32 NumPy GOPS | INT32 QPU GOPS | INT32 Speedup | INT16-in/INT32-acc NumPy GOPS | INT16-in/INT32-acc QPU GOPS | INT16-in/INT32-acc Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1.18 | 6.30 | 5.34x | 1.17 | 6.30 | 5.40x |
| 512 | 0.57 | 16.69 | 29.49x | 0.57 | 16.82 | 29.70x |
| 768 | 0.57 | 20.49 | 36.25x | 0.60 | 20.49 | 33.91x |
| 1024 | 0.21 | 10.73 | 50.15x | 0.23 | 21.67 | 94.38x |

### Operator-Level Performance

Min/max and pooling are reported in GiB/s. Convolution and attention are
reported in GOPS.

| Operator | Setting | NumPy | PyTorch | QPU-C cached | QPU-E execute-only |
| --- | --- | ---: | ---: | ---: | ---: |
| Min/max | INT32, 12 cores | 6.27 | 6.17 | 6.50 | - |
| Min/max | INT16, 12 cores | 6.71 | 6.26 | 6.02 | - |
| AvgPool 2x2 / 2 | INT32, 12 cores | 0.72 | 0.69 | 2.62 | - |
| MaxPool 2x2 / 2 | FP32, 12 cores | 2.66 | 3.03 | 2.65 | - |
| Conv2D | FP32 | 21.69 | 12.74 | 16.25 | 19.87 |
| Conv2D | INT32 | 1.44 | 8.30 | 15.08 | 18.60 |
| Conv2D | INT16-in/INT32-acc | 1.45 | 15.45 | 15.97 | 18.43 |
| Attention core total | FP32 | 82.35 | 5.97 | 12.74 | 14.31 |
| Attention core total | INT32 | 1.13 | 1.79 | 12.81 | 14.41 |

Preliminary end-to-end CNN result: the 12-core execute-only INT32 LeNet-style
pipeline reaches `4.08 GOPS`, compared with `0.83 GOPS` for NumPy, nearly a
`5x` improvement.

## Paper Timeline

| Milestone | Date |
| --- | --- |
| Submission | TBD |
| Acceptance | TBD |
| Camera-ready | TBD |
| Presentation | MLSys 2026 Young Professionals Symposium, Bellevue, WA, 2026 |

## Roadmap

- Broader operator coverage for QPU-first inference.
- CPU/QPU scheduling for mixed workloads.
- A stable persistent runtime API above individual example scripts.
- Lightweight end-to-end LLM inference experiments on Raspberry Pi 5.
- Reproducible benchmark harnesses for cached and execute-only timing.

## Attribution

`qpu-xla` depends on upstream `py-videocore7` rather than vendoring its
implementation. Upstream `py-videocore7` remains owned and licensed by its
authors. The qpu-xla kernels and runtime experiments in this repository are
separate research code associated with the MLSys 2026 YPS paper.
