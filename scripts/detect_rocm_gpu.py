#!/usr/bin/env python3
"""ROCm GPU preflight check for AMD PyTorch wheels.

Detects the AMD GPU "gfx" architecture target (e.g. gfx1150) directly from
the Linux kernel's amdgpu/KFD driver, with no ROCm userspace install
required. Use this before choosing the ``device-gfxNNNN`` extra pinned in
``pyproject.toml`` under ``[project.optional-dependencies].rocm`` (see
``tool.uv.sources`` for the matching index routing), and to sanity-check
that the system is ready for the ROCm pip wheels.

Usage:
    python scripts/detect_rocm_gpu.py

Exit codes:
    0 - at least one usable AMD GPU found, prerequisites look OK
    1 - hard blocker (no amdgpu driver, no KFD GPU nodes, unsupported OS)
    2 - GPU found but one or more soft warnings (permissions, missing libs)
"""

from __future__ import annotations

import ctypes
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
REQUIRED_LIBS = ["libatomic.so.1", "libquadmath.so.0"]
SUPPORTED_PYTHON = {(3, 11), (3, 12), (3, 13), (3, 14)}


@dataclass
class GpuNode:
    node_id: str
    gfx_target_version: int
    simd_count: int

    @property
    def gfx_name(self) -> str:
        """Decode gfx_target_version (major*10000 + minor*100 + step) into
        the AMD gfx architecture name, e.g. 110500 -> 'gfx1150'.

        The step component is hex (e.g. gfx90a has step=10 -> 'a').
        """
        major = self.gfx_target_version // 10000
        minor = (self.gfx_target_version % 10000) // 100
        step = self.gfx_target_version % 100
        return f"gfx{major}{minor}{step:x}"

    @property
    def device_extra(self) -> str:
        """The matching `torch[device-gfxNNNN]` extra name for this GPU."""
        return f"device-{self.gfx_name}"


def read_properties(props_path: Path) -> dict[str, str]:
    props: dict[str, str] = {}
    for line in props_path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            props[parts[0]] = parts[1].strip()
    return props


def find_gpu_nodes() -> list[GpuNode]:
    """Enumerate KFD topology nodes and return actual GPUs (simd_count > 0).

    Node 0 is typically the CPU/APU host node with simd_count == 0.
    """
    nodes: list[GpuNode] = []
    if not KFD_TOPOLOGY.is_dir():
        return nodes
    for node_dir in sorted(KFD_TOPOLOGY.glob("*")):
        props_path = node_dir / "properties"
        if not props_path.exists():
            continue
        props = read_properties(props_path)
        simd_count = int(props.get("simd_count", "0"))
        gfx_target_version = int(props.get("gfx_target_version", "0"))
        if simd_count > 0 and gfx_target_version > 0:
            nodes.append(GpuNode(node_dir.name, gfx_target_version, simd_count))
    return nodes


def check_amdgpu_driver_loaded() -> bool:
    modules = Path("/proc/modules")
    if not modules.exists():
        return False
    return any(line.split()[0] == "amdgpu" for line in modules.read_text().splitlines())


def check_device_access() -> list[str]:
    warnings: list[str] = []
    kfd = Path("/dev/kfd")
    if not kfd.exists():
        warnings.append("/dev/kfd does not exist (amdgpu/KFD driver not exposing compute device)")
    elif not (kfd.stat().st_mode & 0o666):
        warnings.append("/dev/kfd exists but current user may lack read/write access")

    dri = Path("/dev/dri")
    render_nodes = sorted(dri.glob("renderD*")) if dri.exists() else []
    if not render_nodes:
        warnings.append("No /dev/dri/renderD* nodes found")
    return warnings


def check_libs() -> list[str]:
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    return missing


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) in SUPPORTED_PYTHON
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def main() -> int:
    print("== ROCm GPU preflight check ==")
    print(f"OS: {platform.platform()}")

    if platform.system() != "Linux":
        print("BLOCKER: ROCm pip wheels only support Linux (and Windows for some builds).")
        return 1

    if not check_amdgpu_driver_loaded():
        print("BLOCKER: 'amdgpu' kernel module is not loaded. Install/enable the AMD GPU driver first.")
        return 1
    print("OK: amdgpu kernel driver is loaded")

    nodes = find_gpu_nodes()
    if not nodes:
        print("BLOCKER: No AMD GPU compute nodes found under /sys/class/kfd/kfd/topology/nodes.")
        print("         Confirm the GPU is visible to the kernel (e.g. `lspci | grep -i amd`).")
        return 1

    warnings: list[str] = []
    for node in nodes:
        print(
            f"OK: GPU node {node.node_id}: gfx_target_version={node.gfx_target_version} "
            f"-> {node.gfx_name} (simd_count={node.simd_count})"
        )

    if len(nodes) > 1:
        warnings.append(
            "Multiple AMD GPU nodes detected; pick the one you intend to use with "
            "--extra rocm and pin its device-gfxNNNN extra explicitly."
        )

    primary = nodes[0]
    print(f"\nSuggested pyproject.toml extra: {primary.device_extra}")
    print(
        "  e.g. \"torch[{extra}]==<version>+rocm<rocm-ver>\"".format(extra=primary.device_extra)
    )

    warnings.extend(check_device_access())

    missing_libs = check_libs()
    if missing_libs:
        warnings.append(f"Missing shared libraries: {', '.join(missing_libs)}")
    else:
        print("OK: required libraries present (" + ", ".join(REQUIRED_LIBS) + ")")

    py_ok, py_version = check_python_version()
    if py_ok:
        print(f"OK: Python {py_version} is a supported ROCm PyTorch version")
    else:
        warnings.append(
            f"Python {py_version} is outside ROCm's officially supported set (3.11-3.14)"
        )

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
        return 2

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
