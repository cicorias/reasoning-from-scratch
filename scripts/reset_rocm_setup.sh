#!/usr/bin/env bash
# Tidy-up / reset script for the ROCm (AMD GPU) PyTorch setup.
#
# Reverses everything the ROCm setup workflow (see
# ch02/02_setup-tips/rocm-setup.md) could plausibly have touched:
#   1. The project's venv-scoped install (the `rocm` extra's packages).
#   2. Optionally, uv's download cache entries for the AMD ROCm wheels.
#   3. OS-level changes recommended by AMD's docs for OTHER users' machines
#      (render/video group membership, a udev rule file, libatomic/libquadmath
#      system packages) -- these are only DETECTED and reported here, since
#      this script cannot know whether you added them for ROCm specifically
#      or already had them for an unrelated reason.
#
# Usage:
#   scripts/reset_rocm_setup.sh            # venv-scoped reset only (safe default)
#   scripts/reset_rocm_setup.sh --full      # also prune uv's cache for these packages
#   scripts/reset_rocm_setup.sh --dry-run   # print what would happen, change nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --full) FULL=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

echo "== 1. Venv-scoped ROCm install =="
if [ -d ".venv" ]; then
  echo "Re-syncing without the 'rocm' extra (restores the default CPU/CUDA torch build)."
  run uv sync
else
  echo "No .venv found; nothing to reset here."
fi

echo
echo "== 2. uv download cache (AMD ROCm wheels) =="
if [ "$FULL" -eq 1 ]; then
  for pkg in torch torchvision torchaudio triton rocm rocm-bootstrap \
             rocm-sdk-core rocm-sdk-libraries rocm-sdk-device-gfx1150 \
             amd-torch-device-gfx1150 amd-torch-device-gfx115x \
             amd-torchvision-device-gfx1150; do
    run uv cache clean "$pkg" || true
  done
else
  echo "Skipped (pass --full to also prune uv's cache for these packages)."
  echo "Note: this cache is shared across ALL your uv projects, so pruning it"
  echo "only saves disk space; it does not affect other projects' installs."
fi

echo
echo "== 3. OS-level changes (detection only; nothing is removed automatically) =="

echo "-- Group membership --"
CURRENT_GROUPS="$(id -Gn "$USER" 2>/dev/null || true)"
for grp in render video; do
  if echo "$CURRENT_GROUPS" | tr ' ' '\n' | grep -qx "$grp"; then
    echo "You are a member of the '$grp' group."
    echo "  If this was added ONLY for ROCm and you no longer need GPU access, remove it with:"
    echo "    sudo gpasswd -d \"$USER\" $grp"
    echo "  (Log out and back in for the change to take effect.)"
  else
    echo "Not a member of '$grp' (nothing to undo)."
  fi
done

echo
echo "-- udev rule for /dev/kfd, /dev/dri/renderD* permissions --"
UDEV_RULE="/etc/udev/rules.d/70-amdgpu.rules"
if [ -f "$UDEV_RULE" ]; then
  echo "Found $UDEV_RULE."
  echo "  If you (or a setup script) created this file for ROCm, remove it with:"
  echo "    sudo rm $UDEV_RULE"
  echo "    sudo udevadm control --reload-rules && sudo udevadm trigger"
else
  echo "$UDEV_RULE not present (nothing to undo)."
fi

echo
echo "-- System libraries (libatomic, libquadmath) --"
echo "These are common runtime libraries used by many packages beyond ROCm."
echo "Only remove them if you are certain nothing else on this system depends on them, e.g.:"
echo "  Arch/Omarchy:   sudo pacman -R gcc-libs   # provides both; likely a core dependency, avoid removing"
echo "  Debian/Ubuntu:  sudo apt remove libatomic1 libquadmath0"
echo "This script does NOT remove them automatically since they are almost"
echo "always required by other installed software."

echo
echo "Done. Run 'python scripts/detect_rocm_gpu.py' any time to re-check GPU/driver status."
