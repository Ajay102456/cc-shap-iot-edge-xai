# Hard resource-cap runs via Docker (recommended for reported results)

`src/resource_harness.py`'s in-process `memory_cap()` uses
`resource.setrlimit(RLIMIT_AS, ...)`, which caps *virtual address space*.
On this dev machine that cap was observed to be effectively a no-op: NumPy,
LightGBM, and PyTorch already map several hundred MB–1GB+ of virtual
address space at import time (shared libs, BLAS/MKL, CUDA stubs even in
CPU builds), so a 1024 MB tier_pi4 cap can already be exceeded by the
interpreter's baseline footprint before any explainer code runs, causing
`setrlimit` to silently fail (caught) or making the "budget" not reflect
what you'd actually see on a Pi. **This is a documented limitation of the
in-process harness, not a bug to silently paper over** — see
RESEARCH_IMPLEMENTATION_PLAN.md Phase 4.

For results intended to go in the writeup, enforce the cap at the
container level instead, which limits *actual resident memory* (RSS) and
*real* CPU shares, matching what a Pi/Jetson would enforce via cgroups:

```bash
# Build once
docker build -t xai-iot-edge -f docker/Dockerfile .

# Run one resource-tier condition, e.g. tier_pi4 (4 cores, 1GB RAM)
docker run --rm \
  --cpus="4" \
  --memory="1024m" \
  --memory-swap="1024m" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  xai-iot-edge \
  python -m src.run_experiment --data ciciot2023 --n-per-class 5000 --tiers tier_pi4

# tier_jetson_nano (4 cores, 2GB RAM)
docker run --rm --cpus="4" --memory="2048m" --memory-swap="2048m" \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" \
  xai-iot-edge python -m src.run_experiment --data ciciot2023 --tiers tier_jetson_nano
```

If a run gets OOM-killed by Docker (exit code 137), that itself is the
measurement — record it as an OOM event for that (explainer, model, tier)
cell, exactly as `resource_harness.RunResult.oom` records in-process OOMs.
