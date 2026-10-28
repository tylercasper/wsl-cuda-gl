// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#include "mesa_version.h"

#include <cstdio>
#include <cstring>

namespace wslcg::detail {

bool parse_mesa_version(const char* gl_version, int* major, int* minor, int* patch) {
    const char* m = gl_version ? std::strstr(gl_version, "Mesa ") : nullptr;
    if (!m) return false;
    *major = *minor = *patch = 0;
    return std::sscanf(m + 5, "%d.%d.%d", major, minor, patch) >= 2;
}

bool mesa_has_memobj_import_bug(const char* gl_version) {
    int major, minor, patch;
    if (!parse_mesa_version(gl_version, &major, &minor, &patch)) return false;
    if (major != 26) return major < 26;
    return minor == 0 && patch < 3;
}

} // namespace wslcg::detail
