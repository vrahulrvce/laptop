# laptop

Benchmark results for this machine ("laptop") -- raw compute, memory bandwidth,
real local-LLM inference throughput, and a sustained multi-epoch training run
(to catch thermal throttling that a few quick iterations can't reveal).
Purpose: compare against a second PC to decide which one is better suited for
local-model inference, NN training, and continued-pretraining (CPT) work.

## System

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS -- 8 cores / 16 threads, up to 3.8GHz |
| RAM | 32GB DDR5 @ 5600MHz |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU -- 8.6GB VRAM, compute cap 8.9 (Ada Lovelace) |
| Torch | 2.13.0+cu126 |

## Headline numbers

| Benchmark | Result |
|---|---|
| fp16/bf16 matmul (tensor cores) | ~26-28 TFLOPS |
| VRAM bandwidth | ~125-131 GB/s |
| Local LLM inference (Qwen2.5-0.5B-Instruct, eager HF `transformers`) | ~25-26 tokens/sec |
| Sustained 3-minute training run | **no throttling** -- throughput drift -1.3%, peaked at 63C (well below the ~80-87C typical laptop-GPU throttle point), clock speed rose rather than dropped |

Full output: [`results.txt`](results.txt).

## Running it yourself

```
pip install torch transformers
python pc_benchmark.py
```

Self-contained -- only needs `torch` and `transformers`. Downloads
`Qwen/Qwen2.5-0.5B-Instruct` (~1GB) on first run for the inference benchmark.
The sustained-training section runs for a fixed 3-minute wall-clock budget by
default (`duration_sec` in `sustained_training_benchmark`) and logs GPU
temp/clock/power per epoch via `nvidia-smi` if an NVIDIA GPU is present.
