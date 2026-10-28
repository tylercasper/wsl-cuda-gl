// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Compiled with CINTERFACE, so d3d12.h declares the C-style vtable structs, which spell
// out the slot layout explicitly.
#define CINTERFACE
#include <wsl/winadapter.h>
#include <directx/d3d12.h>

#include <cstddef>

size_t c_layout_open_shared_handle_slot() {
    return offsetof(ID3D12DeviceVtbl, OpenSharedHandle) / sizeof(void*);
}
