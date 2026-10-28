#!/usr/bin/env bash
# Copyright 2026 Tyler Casper
# SPDX-License-Identifier: Apache-2.0
#
# Installs the build dependencies on an Ubuntu machine without a GPU (GitHub's hosted
# runners): the CUDA compiler and headers from NVIDIA's apt repository, GL/EGL/GLFW
# headers, Mesa's EGL (for the llvmpipe test), DirectX-Headers and Ninja.
#
#   scripts/ci-install-deps.sh 13.0
set -euo pipefail
cuda_version="${1:?usage: $0 <cuda version, e.g. 13.0>}"
cuda_pkg="${cuda_version/./-}"
. /etc/os-release
distro="ubuntu${VERSION_ID/./}"
sudo=""
[ "$(id -u)" -eq 0 ] || sudo=sudo

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/cuda-keyring.deb" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
$sudo dpkg -i "$tmp/cuda-keyring.deb"
$sudo apt-get update -q
$sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    "cuda-nvcc-${cuda_pkg}" "cuda-cudart-dev-${cuda_pkg}" "cuda-driver-dev-${cuda_pkg}" \
    ninja-build pkg-config libgl-dev libegl-dev libglfw3-dev libegl-mesa0 libgl1-mesa-dri directx-headers-dev

# Make nvcc findable by later steps.
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "/usr/local/cuda-${cuda_version}/bin" >> "$GITHUB_PATH"
fi
