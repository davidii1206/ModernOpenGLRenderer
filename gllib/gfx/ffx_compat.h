#pragma once

// Windows API compatibility shim for FidelityFX FSR2 on Linux
#ifndef _MSC_VER

#ifndef FFX_GCC
#define FFX_GCC
#endif

// FFX headers use size_t without including it
#include <cstddef>
#include <cwchar>

// _countof is MSVC-specific
#ifndef _countof
#define _countof(x) (sizeof(x) / sizeof(x[0]))
#endif

// wcscpy_s (2-arg template form for fixed-size arrays)
template <std::size_t N>
inline void wcscpy_s(wchar_t (&dest)[N], const wchar_t* src) {
    std::wcscpy(dest, src);
}

#endif
