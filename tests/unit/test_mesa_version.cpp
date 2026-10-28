// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// GL_VERSION parsing that decides whether the Mesa < 26.0.3 import workaround applies.

#include "harness.h"
#include "mesa_version.h"

using wslcg::detail::mesa_has_memobj_import_bug;
using wslcg::detail::parse_mesa_version;

static void expect_version(const char* s, int major, int minor, int patch) {
    int a = -1, b = -1, c = -1;
    bool ok = parse_mesa_version(s, &a, &b, &c);
    CHECK(ok);
    if (ok && (a != major || b != minor || c != patch)) {
        std::fprintf(stderr, "  \"%s\" parsed as %d.%d.%d, expected %d.%d.%d\n", s, a, b, c, major, minor, patch);
        ++harness::failures();
    }
}

int main() {
    harness::section("parse_mesa_version");
    expect_version("4.6 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.4", 25, 2, 8);
    expect_version("4.6 (Compatibility Profile) Mesa 26.0.3", 26, 0, 3);
    expect_version("4.6 (Core Profile) Mesa 26.1.0-devel (git-1a2b3c4d5e)", 26, 1, 0);
    expect_version("OpenGL ES 3.1 Mesa 24.0", 24, 0, 0);
    int a, b, c;
    CHECK(!parse_mesa_version("4.6.0 NVIDIA 550.54.14", &a, &b, &c));
    CHECK(!parse_mesa_version("Mesa unknown", &a, &b, &c));
    CHECK(!parse_mesa_version("", &a, &b, &c));
    CHECK(!parse_mesa_version(nullptr, &a, &b, &c));

    harness::section("mesa_has_memobj_import_bug (fixed in 26.0.3)");
    CHECK(mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.4"));
    CHECK(mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 24.0.5"));
    CHECK(mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 26.0.0"));
    CHECK(mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 26.0.2"));
    CHECK(!mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 26.0.3"));
    CHECK(!mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 26.0.10"));
    CHECK(!mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 26.1.0-devel"));
    CHECK(!mesa_has_memobj_import_bug("4.6 (Core Profile) Mesa 27.0.0"));
    CHECK(!mesa_has_memobj_import_bug("4.6.0 NVIDIA 550.54.14"));
    CHECK(!mesa_has_memobj_import_bug(nullptr));
    return harness::finish();
}
