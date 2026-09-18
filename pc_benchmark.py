"""Quick PC-vs-PC benchmark for local-model / NN-training suitability.
Run identically on both machines, compare the printed numbers directly.

Covers the three things that matter for the jobs discussed:
  1. Raw GPU compute (TFLOPS, fp16/bf16 matmul) -- ceiling for everything else
  2. GPU memory bandwidth -- how fast data moves to/from VRAM
  3. Real local-LLM inference throughput (tokens/sec) -- the actual "can I
     run models locally" number, not a proxy
  4. A training step (forward+backward, representative small NN) -- proxy
     for NN-training / CPT workloads, since those are dominated by the
     same matmul+backward pattern regardless of model specifics

Run: python pc_benchmark.py
"""

import platform
import time

import torch

print("=" * 70)
print(f"Machine: {platform.node()} | {platform.system()} {platform.release()}")
print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory/1e9:.1f} GB | compute cap: {props.major}.{props.minor}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)


def bench_matmul(dtype, size=8192, iters=20):
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 2 * size**3 * iters  # matmul FLOPs
    tflops = flops / elapsed / 1e12
    return tflops, elapsed / iters * 1000


def bench_memory_bandwidth(size_mb=512, iters=20):
    n = int(size_mb * 1024 * 1024 / 4)  # float32 elements
    a = torch.randn(n, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b = a.clone()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gb_per_sec = (size_mb / 1024) * 2 * iters / elapsed  # read + write
    return gb_per_sec


def bench_training_step(hidden=2048, layers=6, batch=32, iters=15):
    model = torch.nn.Sequential(
        *[torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.GELU()) for _ in range(layers)]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(batch, hidden, device=device)
    target = torch.randn(batch, hidden, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000  # ms per step


print("\n--- 1. Raw GPU compute (fp32 matmul, 8192x8192) ---")
try:
    tflops32, ms32 = bench_matmul(torch.float32)
    print(f"fp32: {tflops32:.1f} TFLOPS  ({ms32:.1f} ms/iter)")
except RuntimeError as e:
    print(f"fp32 matmul failed (likely OOM at this size): {e}")

if device == "cuda":
    print("\n--- 1b. fp16/bf16 matmul (tensor cores) ---")
    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            tflops, ms = bench_matmul(dtype)
            print(f"{dtype_name}: {tflops:.1f} TFLOPS  ({ms:.1f} ms/iter)")
        except RuntimeError as e:
            print(f"{dtype_name} failed: {e}")

print("\n--- 2. Memory bandwidth ---")
try:
    bw = bench_memory_bandwidth()
    print(f"{'VRAM' if device == 'cuda' else 'RAM'} bandwidth: {bw:.1f} GB/s")
except RuntimeError as e:
    print(f"bandwidth test failed: {e}")

print("\n--- 3. Training step (6-layer MLP, hidden=2048, batch=32) ---")
try:
    ms = bench_training_step()
    print(f"{ms:.2f} ms/step  ({1000/ms:.1f} steps/sec)")
except RuntimeError as e:
    print(f"training step failed: {e}")

print("\n--- 4. Real local-LLM inference throughput ---")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    prompt = "Explain what a neural network is in one paragraph."
    enc = tok(prompt, return_tensors="pt").to(device)

    # warmup
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()

    n_tokens = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=n_tokens, min_new_tokens=n_tokens, do_sample=False, pad_token_id=tok.eos_token_id
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    actual_new = out.shape[1] - enc["input_ids"].shape[1]
    print(f"Qwen2.5-0.5B-Instruct: {actual_new/elapsed:.1f} tokens/sec ({elapsed:.2f}s for {actual_new} tokens)")
except Exception as e:
    print(f"LLM inference benchmark failed: {e!r}")

print("\n--- 5. Sustained multi-epoch training (thermal-throttling check) ---")
print("A few quick iterations can't reveal throttling -- laptops especially")
print("can run fast for 10 seconds then drop 20-40% once they heat up.")
print("Running a real training loop for a fixed wall-clock budget instead,")
print("logging per-epoch speed and (if available) GPU temp/clock/power so")
print("any degradation over time is visible, not just a single burst number.")

import subprocess


def gpu_telemetry():
    if device != "cuda":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        temp, clock, power, util = [x.strip() for x in out.stdout.strip().split(",")]
        return {"temp_c": float(temp), "clock_mhz": float(clock), "power_w": float(power), "util_pct": float(util)}
    except Exception:
        return None


def sustained_training_benchmark(duration_sec=180, hidden=1024, layers=8, batch=128, n_samples=20000, n_classes=20):
    torch.manual_seed(0)
    x_all = torch.randn(n_samples, hidden, device=device)
    y_all = torch.randint(0, n_classes, (n_samples,), device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        *[m for _ in range(layers) for m in (torch.nn.Linear(hidden, hidden), torch.nn.GELU())],
        torch.nn.Linear(hidden, n_classes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_batches = n_samples // batch

    epoch_log = []
    start = time.perf_counter()
    epoch = 0
    while time.perf_counter() - start < duration_sec:
        epoch += 1
        perm = torch.randperm(n_samples, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter()
        total_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch : (i + 1) * batch]
            xb, yb = x_all[idx], y_all[idx]
            opt.zero_grad()
            out = model(xb)
            loss = torch.nn.functional.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t_epoch
        telemetry = gpu_telemetry()
        epoch_log.append({
            "epoch": epoch, "time_s": epoch_time, "loss": total_loss / n_batches,
            "samples_per_sec": n_samples / epoch_time, **({"telemetry": telemetry} if telemetry else {}),
        })
        t = telemetry
        t_str = f"  temp={t['temp_c']:.0f}C clock={t['clock_mhz']:.0f}MHz power={t['power_w']:.0f}W" if t else ""
        print(f"  epoch {epoch:>3}  {epoch_time*1000:>7.1f} ms  loss={total_loss/n_batches:.3f}  "
              f"{n_samples/epoch_time:>8.0f} samples/sec{t_str}")

    return epoch_log


try:
    log = sustained_training_benchmark(duration_sec=180)
    n = len(log)
    if n >= 4:
        first_quarter = log[: max(1, n // 4)]
        last_quarter = log[-max(1, n // 4):]
        avg_first = sum(e["samples_per_sec"] for e in first_quarter) / len(first_quarter)
        avg_last = sum(e["samples_per_sec"] for e in last_quarter) / len(last_quarter)
        drift_pct = (avg_last - avg_first) / avg_first * 100
        print(f"\n  {n} epochs completed in the time budget.")
        print(f"  first-quarter avg: {avg_first:.0f} samples/sec")
        print(f"  last-quarter avg:  {avg_last:.0f} samples/sec")
        print(f"  drift: {drift_pct:+.1f}%  "
              f"({'THROTTLING -- sustained perf is meaningfully lower than burst' if drift_pct < -10 else 'stable, no meaningful throttling detected'})")
        if log[0].get("telemetry") and log[-1].get("telemetry"):
            t0, t1 = log[0]["telemetry"], log[-1]["telemetry"]
            print(f"  GPU temp: {t0['temp_c']:.0f}C -> {t1['temp_c']:.0f}C  "
                  f"clock: {t0['clock_mhz']:.0f}MHz -> {t1['clock_mhz']:.0f}MHz  "
                  f"power: {t0['power_w']:.0f}W -> {t1['power_w']:.0f}W")
    else:
        print(f"  only {n} epoch(s) fit in the time budget -- inconclusive for throttling, "
              f"increase duration_sec or shrink the model/batch")
except RuntimeError as e:
    print(f"sustained training benchmark failed: {e}")

print("\n" + "=" * 70)
print("Done. Run this same script on the other PC and compare.")
