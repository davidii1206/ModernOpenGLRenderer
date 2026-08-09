#pragma once
#include <cstddef>
#include <cstdint>
#include <glad/glad.h>

#include <ffx_fsr2.h>
#include <ffx_fsr2_interface.h>

struct FfxFsr2Interface;
struct FfxFsr2Context;
struct FfxResource;

// Backend size query + interface populator
size_t      ffxFsr2GetScratchMemorySizeGL();
FfxErrorCode ffxFsr2GetInterfaceGL(FfxFsr2Interface* outInterface, void* scratchBuffer, size_t scratchBufferSize);

// Resource helpers for the application
FfxResource     ffxGetTextureResourceGL(GLuint textureGL, uint32_t width, uint32_t height, GLenum imgFormat, const wchar_t* name = nullptr);
FfxResource     ffxGetBufferResourceGL(GLuint bufferGL, uint32_t size, const wchar_t* name = nullptr);
GLuint          ffxGetGLImage(FfxFsr2Context* context, uint32_t resId);
FfxSurfaceFormat ffxGetSurfaceFormatGL(GLenum fmt);
