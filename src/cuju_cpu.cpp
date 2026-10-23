// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// `libcuju_cpu.so`, the C API of `environment_api.h` over the CPU batch.
#include "cuju/cpu/batch_environment.h"

using cuju::cpu::BatchEnvironment;
#define CUJU_BACKEND_NAME "cpu"

#include "environment_api.h"
#include "scenario_api.h"
#include "state_api.h"

// MSVC exports what `cuju.def` lists, from the C API headers, and this function besides.
#if defined(_MSC_VER)
#define CUJU_CPU_EXPORT __declspec(dllexport)
#else
#define CUJU_CPU_EXPORT
#pragma GCC visibility push(default)
#endif
extern "C" {

// The processors that the CPU quota of the process's cgroup allows, rounded up, 0 without a quota
// or -1 on a failure. `threads=0` takes no more. `root` prefixes the paths it reads under `/proc`
// and `/sys`, or null for none.
CUJU_CPU_EXPORT int cuju_cpu_quota(const char *root) {
    try {
        return cuju::cpu::QuotaProcessors(root ? root : "");
    } catch (const std::exception &exception) {
        return Fail(exception);
    }
}

}  // extern "C"
#if !defined(_MSC_VER)
#pragma GCC visibility pop
#endif
