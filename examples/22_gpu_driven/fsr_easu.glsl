#version 460 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define A_GPU 1
#define A_GLSL 1

#include "ffx_a.h"

#define FSR_EASU_F 1

layout(location = 0) uniform uvec4 FSR_EASU_con0;
layout(location = 1) uniform uvec4 FSR_EASU_con1;
layout(location = 2) uniform uvec4 FSR_EASU_con2;
layout(location = 3) uniform uvec4 FSR_EASU_con3;

layout(binding = 0) uniform sampler2D fsr_input;
layout(binding = 1, rgba16f) writeonly uniform image2D fsr_output;

AF4 FsrEasuRF(AF2 p) { return AF4(textureGather(fsr_input, p, 0)); }
AF4 FsrEasuGF(AF2 p) { return AF4(textureGather(fsr_input, p, 1)); }
AF4 FsrEasuBF(AF2 p) { return AF4(textureGather(fsr_input, p, 2)); }

#include "ffx_fsr1.h"

void main() {
    AU2 gxy = AU2(gl_GlobalInvocationID.xy);
    AF3 color;
    FsrEasuF(color, gxy, FSR_EASU_con0, FSR_EASU_con1, FSR_EASU_con2, FSR_EASU_con3);
    imageStore(fsr_output, ivec2(gxy), AF4(color, 1.0));
}
