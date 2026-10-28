// Copyright 2026 Tyler Casper
// SPDX-License-Identifier: Apache-2.0

// Compile-time checks on the public header: it is self-contained, and its API has the
// shape the README promises. Built as C++17 and again as C++20; see also
// public_header_after_gl_cuda.cpp.

#include "wsl_cuda_gl.h"

#include <type_traits>

static_assert(WSLCG_VERSION_MAJOR >= 0 && WSLCG_VERSION_MINOR >= 0 && WSLCG_VERSION_PATCH >= 0);
static_assert(std::is_base_of_v<std::runtime_error, wslcg::Error>);
static_assert(!std::is_copy_constructible_v<wslcg::Buffer> && !std::is_copy_assignable_v<wslcg::Buffer>);
static_assert(std::is_nothrow_move_constructible_v<wslcg::Buffer>);
static_assert(std::is_nothrow_move_assignable_v<wslcg::Buffer>);
static_assert(!std::is_convertible_v<std::size_t, wslcg::Buffer>, "Buffer(size_t) must be explicit");
static_assert(std::is_same_v<decltype(std::declval<wslcg::Buffer&>().map()), void*>);
static_assert(std::is_same_v<decltype(std::declval<const wslcg::Buffer&>().gl_buffer()), unsigned int>);
static_assert(noexcept(std::declval<const wslcg::Buffer&>().size()));

bool header_compiles_after_gl_and_cuda();

int main() { return header_compiles_after_gl_and_cuda() ? 0 : 1; }
