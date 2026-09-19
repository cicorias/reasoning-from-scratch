# Using an AMD GPU (ROCm) Instead of NVIDIA/CUDA

This project's default `uv sync` installs a CPU/CUDA-appropriate PyTorch build automatically. If you have an **AMD GPU** on Linux, uv has no way to auto-detect that (Python packaging environment markers only cover OS/architecture, not GPU vendor), so ROCm support is provided as an **opt-in extra** instead, sourced entirely as prebuilt wheels from AMD's own package index (no source builds).

This has been verified end-to-end on an AMD Radeon 880M/890M (Ryzen AI APU, `gfx1150`) on Arch-based Linux (Omarchy), producing a working `torch.cuda.is_available() == True` (PyTorch's ROCm builds reuse the CUDA API namespace) and a successful GPU matrix multiply.

&nbsp;

## Overview: the "pre-setup" scripts

Two scripts support this workflow:

| Script | Purpose |
| --- | --- |
| `scripts/detect_rocm_gpu.py` | **Preflight check.** Detects your AMD GPU's `gfx` architecture target and validates driver/permissions/library prerequisites, *before* you install anything. Read-only, no ROCm install required. |
| `scripts/reset_rocm_setup.sh` | **Teardown / reset.** Reverses the venv-scoped ROCm install and reports any OS-level changes (group membership, udev rules) that may need manual cleanup. |

&nbsp;

## 1. Run the preflight check

From the repository root:

```bash
python scripts/detect_rocm_gpu.py
```

This script reads the Linux kernel's `amdgpu`/KFD driver directly (no ROCm userspace install required) and reports:

- Whether the `amdgpu` kernel driver is loaded
- Each detected AMD GPU's **gfx architecture target** (e.g. `gfx1150`, `gfx1100`, `gfx942`), decoded from `/sys/class/kfd/kfd/topology/nodes/*/properties`
- Whether `/dev/kfd` and `/dev/dri/renderD*` are accessible (either directly world-accessible, or via `render`/`video` group membership -- see [Prerequisites](#prerequisites-for-other-machines) below if not)
- Whether required libraries (`libatomic`, `libquadmath`) are present
- Whether your Python version is in ROCm's officially supported range (3.11-3.14; see the [Python version caveat](#python-version-caveat) below for this project specifically)

Example output on a Ryzen AI APU with an integrated Radeon 880M/890M GPU:

```
OK: GPU node 1: gfx_target_version=110500 -> gfx1150 (simd_count=32)
Suggested pyproject.toml extra: device-gfx1150
```

Exit code `0` means all checks passed; `1` means a hard blocker (no driver, no GPU node, unsupported OS); `2` means the GPU was found but there are warnings to review (permissions, missing libs).

&nbsp;

## 2. Match the extra in `pyproject.toml`

The `[project.optional-dependencies].rocm` group in `pyproject.toml` pins `torch`, `torchvision`, and `torchaudio` (plus their transitive ROCm SDK dependencies -- see [How this works](#how-this-works) below) to a specific `device-gfxNNNN` extra and ROCm/PyTorch version. If the preflight script suggests a different `gfx` target than what's currently pinned, update every `device-gfx1150` occurrence in the `rocm = [...]` list and `[tool.uv.sources]` to match your GPU (or use the broader `device-all` extra, which is larger but works across GPU families -- see AMD's index for available `rocm-sdk-device-*` package names).

&nbsp;

## 3. Create a Python < 3.14 venv, then install

<a id="python-version-caveat"></a>

**Python version caveat:** the ROCm extra requires **Python < 3.14**. This isn't an AMD limitation (their index does publish `cp314` wheels) -- it works around a `uv` universal-lock resolver limitation where marker-forks spanning Python 3.14 and non-Linux platforms merge incorrectly and produce spurious "no solution found" errors. If your default `uv python` is 3.14 (check with `uv run python --version`), create the project venv with an older interpreter first:

```bash
uv venv --python 3.12 .venv
```

Then install:

```bash
uv sync --extra rocm
```

This downloads several GB of wheels (`torch`, `triton`, `rocm-sdk-core`, `rocm-sdk-device-gfx1150`, etc.) -- expect this to take a while on the first run. Everything is installed into `.venv` only; **no OS packages are touched**.

&nbsp;

## 4. Verify

```bash
uv run python -c "
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
print((x @ x).sum().item())
"
```

Expected output (device name will match your GPU):

```
2.13.0+rocm10.0.0 True
AMD Radeon 890M Graphics
<some number>
```

&nbsp;

## How this works

AMD publishes ROCm-enabled PyTorch wheels across **two separate simple package indexes**:

- `https://stable.repo.amd.com/rocm/pytorch/whl-next/` -- `torch`, `torchvision`, `torchaudio`, `triton`, and the per-GPU `amd-torch-device-gfxNNNN` packages.
- `https://stable.repo.amd.com/rocm/core/whl-next/` -- the ROCm SDK itself: `rocm`, `rocm-bootstrap`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfxNNNN`.

Both are configured as `explicit = true` indexes in `[[tool.uv.index]]`, routed via `[tool.uv.sources]`, gated to Linux/non-aarch64/Python<3.14 by environment markers. Two non-obvious details drove the final working configuration:

1. **`tool.uv.sources` only applies to *direct* project dependencies.** `torch` transitively depends on `rocm[libraries]`, which transitively depends on `rocm-sdk-libraries`, etc. -- but uv silently ignores source routing for anything not listed directly in `pyproject.toml`. That's why every package in that dependency chain (`rocm`, `rocm-bootstrap`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1150`, `triton`, `amd-torch-device-gfx1150`, `amd-torch-device-gfx115x`, `amd-torchvision-device-gfx1150`) is listed explicitly in the `rocm` extra, pinned to the exact version AMD publishes (important, since an unrelated package named `rocm-bootstrap` also exists on PyPI).
2. **AMD's index wheels don't publish upload-date metadata.** This project sets a repo-wide `exclude-newer = "7 days"` freshness cutoff (`[tool.uv]`), which by default excludes any wheel uv can't date -- and no date *value* override fixes this (uv still can't prove such a wheel is "old enough"). The fix is `exclude-newer-package = { torch = false, ... }`: the boolean `false` fully disables the cutoff for that package name, rather than substituting a different date.

If you'd rather use PyTorch.org's own official `rocm7.x` channel instead of AMD's index: as of this writing, that channel publishes only one `torch` build per ROCm version, whose `rocm` stub dependency is published there as a **source-only tarball** (no wheel), which breaks a wheels-only install. AMD's own index was used instead for that reason.

&nbsp;

## Tearing down / resetting

To undo the ROCm setup:

```bash
scripts/reset_rocm_setup.sh          # re-syncs the venv without the rocm extra
scripts/reset_rocm_setup.sh --full   # also prunes uv's cache of the AMD wheels
scripts/reset_rocm_setup.sh --dry-run  # preview only, changes nothing
```

This script re-runs `uv sync` (without `--extra rocm`) to restore the default CPU/CUDA torch build in `.venv`, and reports (without automatically changing) any OS-level state the general ROCm setup process can involve on other machines: `render`/`video` group membership, a `/etc/udev/rules.d/70-amdgpu.rules` file, and `libatomic`/`libquadmath` system packages. On this project's own verified dev machine, none of these OS-level changes were actually necessary (see below), so there was nothing to remove there -- but the script still checks and reports, since your machine may differ.

&nbsp;

## Prerequisites for other machines

<a id="prerequisites-for-other-machines"></a>

AMD's official prerequisite docs are validated against Ubuntu/RHEL/SLES; other distros (e.g. Arch-based) aren't officially tested, though the pip wheels bundle their own ROCm runtime libraries and are generally distro-agnostic as long as the following are true:

- The `amdgpu` kernel driver is loaded (`lsmod | grep amdgpu`).
- Your user can access `/dev/kfd` and `/dev/dri/renderD*` -- either because they're world-accessible, or because your user is in the `render` and/or `video` group (`sudo usermod -aG render,video "$USER"`, then log out/in), or via a udev rule granting access.
- `libatomic` and `libquadmath` are installed (commonly part of your distro's base GCC runtime libraries already).
- A supported Python version (3.11-3.14 per AMD; this project's `rocm` extra additionally requires **< 3.14**, see above).

`scripts/detect_rocm_gpu.py` checks all of these for you.

&nbsp;

## Questions?

If you have any questions, please don't hesitate to reach out via the [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) forum in this GitHub repository.
