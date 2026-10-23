# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

# `native` needs a GPU on the build machine; a build without one names the architectures, for
# example with `-DCMAKE_CUDA_ARCHITECTURES="86;89"`.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The version has one source, `__version__` in `python/cuju/_version.py`, which the package
# metadata reads as well.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/python/cuju/_version.py" cuju_version
    REGEX "^__version__ = ")
string(REGEX REPLACE "^__version__ = ['\"]([^'\"]+)['\"].*$" "\\1" cuju_version "${cuju_version}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" cuju_version_number "${cuju_version}")
project(cuju VERSION ${cuju_version_number} LANGUAGES CXX)

set(CUJU_CUDA AUTO CACHE STRING "Build the CUDA backend, libcuju_cuda.so: AUTO, ON or OFF")
set_property(CACHE CUJU_CUDA PROPERTY STRINGS AUTO ON OFF)
option(CUJU_CPU "Build the CPU backend, libcuju_cpu.so" ON)

# Windows builds with MSVC, whose CPU and CUDA libraries give the same bits.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Exact: before Visual Studio 2022 (MSVC 19.30), /fp:precise could contract.
    if(NOT MSVC OR MSVC_VERSION LESS 1930)
        message(FATAL_ERROR "cuju builds on Windows with the MSVC of Visual Studio 2022 or later")
    endif()
    # The static C runtime, so that the DLL needs no msvcp140.dll beside it.
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
elseif(APPLE)
    # macOS builds the CPU library only, with Apple's Clang or another Clang.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        message(FATAL_ERROR "cuju builds on macOS with Clang, found `${CMAKE_CXX_COMPILER_ID}`")
    endif()
    if(CUJU_CUDA STREQUAL "ON")
        message(FATAL_ERROR "cuju has no CUDA backend on macOS; pass -DCUJU_CUDA=OFF")
    endif()
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "cuju builds only on Linux, Windows and macOS, found "
        "`${CMAKE_SYSTEM_NAME}`")
endif()
# GRF's results come from x86-64. aarch64 computes the same IEEE 754 operations, and where the two
# differ, in float to integer conversions and in the floating point environment, the code follows
# x86-64 explicitly. On macOS the target is the one architecture that CMAKE_OSX_ARCHITECTURES
# names, else the host's.
set(cuju_processor ${CMAKE_SYSTEM_PROCESSOR})
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES cuju_architecture_count)
    if(NOT cuju_architecture_count EQUAL 1)
        message(FATAL_ERROR "cuju builds for one architecture at a time, found "
            "`${CMAKE_OSX_ARCHITECTURES}`")
    endif()
    set(cuju_processor ${CMAKE_OSX_ARCHITECTURES})
endif()
if(cuju_processor MATCHES "^(x86_64|AMD64|amd64)$")
    set(cuju_x86_64 ON)
elseif(cuju_processor MATCHES "^(aarch64|arm64|ARM64)$")
    set(cuju_x86_64 OFF)
else()
    message(FATAL_ERROR "cuju builds only on x86-64 and aarch64 hosts, found `${cuju_processor}`")
endif()
if(MSVC AND NOT cuju_x86_64)
    message(FATAL_ERROR "cuju builds on Windows for x86-64 only")
endif()

if(CUJU_CUDA STREQUAL "AUTO" AND APPLE)
    set(cuju_build_cuda OFF)
elseif(CUJU_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(cuju_build_cuda ON)
    else()
        set(cuju_build_cuda OFF)
        message(STATUS "cuju: no CUDA compiler found, so the build skips the CUDA backend")
    endif()
elseif(CUJU_CUDA)
    set(cuju_build_cuda ON)
else()
    set(cuju_build_cuda OFF)
endif()
if(cuju_build_cuda)
    enable_language(CUDA)
endif()

# The compute capabilities whose digests were checked on a GPU of their own, which the wheel may
# hold. Another one, such as 7.0 for the V100, builds on request and must pass `pytest tests` on
# such a GPU before its results are trusted.
set(CUJU_CHECKED_CUDA_ARCHITECTURES 75 80 86 89)
if(cuju_build_cuda)
    set(cuju_architectures ${CMAKE_CUDA_ARCHITECTURES})
    if(cuju_architectures STREQUAL "native" AND DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(cuju_architectures ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    foreach(architecture IN LISTS cuju_architectures)
        string(REGEX REPLACE "-(real|virtual)$" "" capability "${architecture}")
        if(capability MATCHES "^[0-9]+$" AND NOT capability IN_LIST CUJU_CHECKED_CUDA_ARCHITECTURES)
            message(WARNING "cuju: no check has run on a GPU of compute capability ${capability} "
                "yet; run `pytest tests` on one before trusting this build")
        endif()
    endforeach()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The build types add no flags, so none changes a floating-point result. Keep it that way:
# exactness depends on it.
foreach(type DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_CXX_FLAGS_${type} "")
    set(CMAKE_CUDA_FLAGS_${type} "")
endforeach()

# The results match GRF bit for bit only with the following flags: no contraction into a fused
# multiply-add, IEEE 754 division and square root, subnormals kept. ptxas -O1 changes no result
# and runs 2.8% to 4.3% faster on the RTX 4090 than its default -O3. MSVC's /fp:precise keeps
# IEEE 754 semantics and, from Visual Studio 2022, contracts nothing: /fp:contract stays off, and
# the build refuses it, as it refuses /fp:fast. The expected digests guard both.
if(MSVC)
    set(cuju_no_contraction /fp:precise)
    set(cuju_cxx_exact_flags /O2 /fp:precise /EHsc /permissive- /Zc:__cplusplus)
    set(cuju_host_exact_flags -Xcompiler=/fp:precise,/EHsc,/permissive-,/Zc:__cplusplus)
else()
    set(cuju_no_contraction -ffp-contract=off)
    set(cuju_cxx_exact_flags -O2 -ffp-contract=off)
    set(cuju_host_exact_flags -Xcompiler=-ffp-contract=off)
endif()
add_library(cuju_exact_flags INTERFACE)
target_compile_options(cuju_exact_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cuju_cxx_exact_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -Xptxas=-O1 --fmad=false -prec-div=true -prec-sqrt=true
        -ftz=false --expt-relaxed-constexpr --extended-lambda
        ${cuju_host_exact_flags}>)
# The libm ports compute each fused step of glibc's FMA build with `fma`, which is correctly
# rounded everywhere; -mfma makes it one instruction on x86-64, as it is on aarch64 from the start.
if(cuju_x86_64 AND NOT MSVC)
    target_compile_options(cuju_exact_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-mfma> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mfma>)
endif()
target_include_directories(cuju_exact_flags INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Reproducible builds: no directory of the build machine enters a library, and nvcc names the
# intermediate files that it embeds after the object file rather than after its process. Neither
# changes an instruction. The later map wins, so the build directory may sit inside the source.
if(NOT MSVC)
    target_compile_options(cuju_exact_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${PROJECT_SOURCE_DIR}/=
            -ffile-prefix-map=${PROJECT_BINARY_DIR}/=>
        $<$<COMPILE_LANGUAGE:CUDA>:-objtemp -Xcompiler=-ffile-prefix-map=${PROJECT_SOURCE_DIR}/=
            -Xcompiler=-ffile-prefix-map=${PROJECT_BINARY_DIR}/=>)
else()
    # MSVC's /Brepro replaces the time in the objects and the DLL with a hash of their contents.
    target_compile_options(cuju_exact_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/Brepro>
        $<$<COMPILE_LANGUAGE:CUDA>:-objtemp -Xcompiler=/Brepro>)
    target_link_options(cuju_exact_flags INTERFACE LINKER:/Brepro)
endif()

# A build of 5 holds only the 5 versus 5 scenarios, and its worlds take half the memory.
set(CUJU_TEAM_PLAYERS 11 CACHE STRING "The room for players in one team")
target_compile_definitions(cuju_exact_flags INTERFACE CUJU_TEAM_PLAYERS=${CUJU_TEAM_PLAYERS})

add_library(cuju_state STATIC src/environment_state.cpp)
target_link_libraries(cuju_state PUBLIC cuju_exact_flags)

if(cuju_build_cuda)
    add_library(cuju_cuda SHARED src/cuju_cuda.cu)
    target_compile_definitions(cuju_cuda PRIVATE CUJU_WARP_ENVIRONMENT)
    target_link_libraries(cuju_cuda PRIVATE cuju_state)
endif()

# None of the following flags changes a result: none contracts or reorders an operation, and -O3
# vectorizes only loops that keep their order. `x86-64-v3` needs AVX2, BMI2 and FMA (Intel Haswell,
# AMD Excavator and later), as MSVC's /arch:AVX2 does; aarch64 builds for the base Armv8-A. Any
# other flag, such as `-march=native`, must pass `pytest tests`.
if(MSVC)
    set(cuju_default_cpu_flags "/O2 /arch:AVX2")
elseif(cuju_x86_64)
    set(cuju_default_cpu_flags "-O3 -march=x86-64-v3 -fno-math-errno -fno-semantic-interposition")
else()
    set(cuju_default_cpu_flags "-O3 -fno-math-errno -fno-semantic-interposition")
endif()
# A Mach-O library has no semantic interposition, and Apple's Clang warns about the flag.
if(APPLE)
    string(REPLACE " -fno-semantic-interposition" "" cuju_default_cpu_flags
        "${cuju_default_cpu_flags}")
endif()
set(CUJU_CPU_FLAGS "${cuju_default_cpu_flags}"
        CACHE STRING "The flags of the CPU backend, which must keep every result bit for bit")
separate_arguments(cuju_effective_flags UNIX_COMMAND "${CUJU_CPU_FLAGS} ${CMAKE_CXX_FLAGS}")
foreach(flag IN LISTS cuju_effective_flags)
    if(flag MATCHES "^-(Ofast|ffast-math|funsafe-math-optimizations|fassociative-math|freciprocal-math|ffinite-math-only|fno-signed-zeros|fcx-limited-range|ffp-contract=(fast|on)|ffp-model=(fast|aggressive)|mfpmath=387|mno-sse2|mlow-precision-(recip-sqrt|sqrt|div))$")
        message(FATAL_ERROR "the flag `${flag}` can change a floating-point result, "
            "which every output must keep bit for bit")
    endif()
    if(MSVC AND flag MATCHES "^[-/](fp:fast|fp:contract|Qfast_transcendentals)$")
        message(FATAL_ERROR "the flag `${flag}` can change a floating-point result, "
            "which every output must keep bit for bit")
    endif()
endforeach()
# The host compiler's flags count too: GCC's and Clang's contraction, MSVC's /fp:fast and
# /fp:contract.
string(JOIN "|" cuju_inexact_cuda_flags use_fast_math ftz=true prec-div=false prec-sqrt=false
    fmad=true "ffp-contract=(fast|on)" "fp:(fast|contract)")
if(CMAKE_CUDA_FLAGS MATCHES "${cuju_inexact_cuda_flags}")
    message(FATAL_ERROR "CMAKE_CUDA_FLAGS `${CMAKE_CUDA_FLAGS}` can change a floating-point result")
endif()
if(CUJU_CPU)
    find_package(Threads REQUIRED)
    add_library(cuju_cpu SHARED src/cuju_cpu.cpp)
    separate_arguments(cuju_cpu_flags UNIX_COMMAND "${CUJU_CPU_FLAGS}")
    # Source options follow the target's, so `CUJU_CPU_FLAGS` overrides the -O2 of
    # `cuju_exact_flags`, and the trailing -ffp-contract=off (/fp:precise) keeps the contraction off
    # after it.
    set_source_files_properties(src/cuju_cpu.cpp PROPERTIES
        COMPILE_OPTIONS "${cuju_cpu_flags};${cuju_no_contraction}")
    if(NOT MSVC)
        target_compile_options(cuju_cpu PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    endif()
    target_link_libraries(cuju_cpu PRIVATE cuju_state Threads::Threads)
endif()

# A DLL exports what a `.def` file lists: each `cuju_` function whose definition starts a line of
# the C API headers, as `#pragma GCC visibility` exports them elsewhere.
if(WIN32)
    file(GLOB cuju_api_headers ${PROJECT_SOURCE_DIR}/src/*_api.h)
    list(SORT cuju_api_headers)
    set(cuju_exports "")
    foreach(header IN LISTS cuju_api_headers)
        file(STRINGS ${header} lines REGEX "^[A-Za-z][^(]*[ *]cuju_[a-z0-9_]+\\(")
        # A `;` splits a line in two, and only the part that starts the line can name a definition.
        foreach(line IN LISTS lines)
            if(line MATCHES "^[A-Za-z][^(]*[ *](cuju_[a-z0-9_]+)\\(")
                list(APPEND cuju_exports ${CMAKE_MATCH_1})
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES cuju_exports)
    list(SORT cuju_exports)
    list(JOIN cuju_exports "\n    " cuju_exports)
    file(CONFIGURE OUTPUT ${PROJECT_BINARY_DIR}/cuju.def
        CONTENT "EXPORTS\n    ${cuju_exports}\n" @ONLY)
    foreach(target cuju_cpu cuju_cuda)
        if(TARGET ${target})
            target_sources(${target} PRIVATE ${PROJECT_BINARY_DIR}/cuju.def)
        endif()
    endforeach()
endif()

# A DLL is a runtime artifact; its import library stays out of the wheel.
if(SKBUILD)
    if(cuju_build_cuda)
        install(TARGETS cuju_cuda LIBRARY DESTINATION cuju RUNTIME DESTINATION cuju
            ARCHIVE DESTINATION cuju EXCLUDE_FROM_ALL)
    endif()
    if(CUJU_CPU)
        install(TARGETS cuju_cpu LIBRARY DESTINATION cuju RUNTIME DESTINATION cuju
            ARCHIVE DESTINATION cuju EXCLUDE_FROM_ALL)
    endif()
endif()

# NVTX ranges around the calls of the C API, a profiling aid for Nsight Systems. NVTX 3 is
# header-only and comes with the CUDA toolkit. The option is off, so a default build has no ranges,
# and the kernels are the same either way.
option(CUJU_NVTX "NVTX ranges around the calls of the C API" OFF)
if(CUJU_NVTX)
    find_path(CUJU_NVTX_INCLUDE nvtx3/nvToolsExt.h
        HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} /usr/local/cuda/include)
    if(CUJU_NVTX_INCLUDE)
        foreach(library cuju_cuda cuju_cpu)
            if(TARGET ${library})
                target_compile_definitions(${library} PRIVATE CUJU_NVTX)
                # An implicit directory such as /usr/include, named again, breaks `#include_next`.
                if(NOT CUJU_NVTX_INCLUDE IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
                    target_include_directories(${library} SYSTEM PRIVATE ${CUJU_NVTX_INCLUDE})
                endif()
                # NVTX loads an attached tool with `dlopen`.
                target_link_libraries(${library} PRIVATE ${CMAKE_DL_LIBS})
            endif()
        endforeach()
    else()
        message(WARNING "cuju: CUJU_NVTX found no NVTX 3 headers, so the libraries have no ranges")
    endif()
endif()
