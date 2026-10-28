// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace wslcg::detail {

// Extracts X.Y.Z from a GL_VERSION string such as
// "4.6 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.4". False if it isn't Mesa's.
// A missing patch number reads as 0.
bool parse_mesa_version(const char* gl_version, int* major, int* minor, int* patch);

// Whether this Mesa's d3d12 driver has the GL_EXT_memory_object import bug fixed
// in Mesa 26.0.3 (see MesaMemobjImportFix in d3d12_bridge.h).
bool mesa_has_memobj_import_bug(const char* gl_version);

} // namespace wslcg::detail
