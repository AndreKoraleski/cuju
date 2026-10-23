// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// The paths that the C API takes, which are UTF-8 on every system.
#pragma once

#include <string>

#if defined(_WIN32)
#include <filesystem>
#endif

namespace cuju {

#if defined(_WIN32)
// Windows reads a narrow path in the ANSI code page, so the path opens as UTF-16 instead.
inline std::filesystem::path FilePath(const std::string &utf8) {
    return std::filesystem::u8path(utf8);
}
#else
inline const std::string &FilePath(const std::string &path) { return path; }
#endif

}  // namespace cuju
