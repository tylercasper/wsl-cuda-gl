// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// The Mesa workaround patches ID3D12Device::OpenSharedHandle's vtable slot, whose index
// it derives from the C++ header via the Itanium pointer-to-member ABI. Cross-check
// that against the C-style vtable layout declared by the same header.

#include "d3d12_bridge.h"
#include "harness.h"

size_t c_layout_open_shared_handle_slot();

int main() {
    size_t from_abi = wslcg::detail::open_shared_handle_slot();
    size_t from_c_layout = c_layout_open_shared_handle_slot();
    std::printf("OpenSharedHandle slot: %zu (pointer-to-member ABI), %zu (C vtable layout)\n", from_abi,
                from_c_layout);
    CHECK(from_abi == from_c_layout);
    return harness::finish();
}
