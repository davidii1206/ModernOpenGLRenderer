// Wrapper for the FidelityFX FSR2 CPU API source with Linux compatibility.
// This file exists to force-include the compat shim before the FSR2 source.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
// MIT license — see ffx_fsr2.cpp for full license text.

#include "ffx_compat.h"
#include <ffx_fsr2.cpp>
