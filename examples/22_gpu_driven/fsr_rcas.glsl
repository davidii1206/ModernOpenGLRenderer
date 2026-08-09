#version 460 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define A_GPU 1
#define A_GLSL 1

#include "ffx_a.h"

#define FSR_RCAS_F 1

layout(location = 0) uniform uvec4 FSR_RCAS_con;

layout(binding = 0) uniform sampler2D fsr_input;
layout(binding = 1, rgba16f) writeonly uniform image2D fsr_output;

AF4 FsrRcasLoadF(ASU2 p) { return AF4(texelFetch(fsr_input, ivec2(p), 0)); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}

#include "ffx_fsr1.h"

void main() {
    AU2 gxy = AU2(gl_GlobalInvocationID.xy);
    AF1 r, g, b;
    FsrRcasF(r, g, b, gxy, FSR_RCAS_con);
    imageStore(fsr_output, ivec2(gxy), AF4(r, g, b, 1.0));
}
