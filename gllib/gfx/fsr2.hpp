#pragma once

#include <cstddef>
#include <cstdint>
#include <glad/glad.h>
#include <ffx_fsr2.h>

namespace gfx {

struct Fsr2InitParams {
    uint32_t renderWidth  = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth  = 0;
    uint32_t displayHeight = 0;
    uint32_t flags = FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE |
                     FFX_FSR2_ENABLE_DEPTH_INVERTED |
                     FFX_FSR2_ENABLE_AUTO_EXPOSURE;
};

struct Fsr2DispatchParams {
    GLuint color         = 0;
    GLuint depth         = 0;
    GLuint motionVectors = 0;
    GLuint output        = 0;
    float  jitterX       = 0.0f;
    float  jitterY       = 0.0f;
    float  sharpness     = 0.5f;
    float  deltaTime     = 16.0f;
    float  nearPlane     = 0.1f;
    float  farPlane      = 500.0f;
    float  fovVerticalRad = 1.047f;
    bool   reset         = false;
};

class Fsr2 {
public:
    Fsr2();
    ~Fsr2();

    Fsr2(const Fsr2&) = delete;
    Fsr2& operator=(const Fsr2&) = delete;
    Fsr2(Fsr2&&) = delete;
    Fsr2& operator=(Fsr2&&) = delete;

    bool init(const Fsr2InitParams& params);
    void dispatch(const Fsr2DispatchParams& params);
    void shutdown();

    bool valid() const { return initialized_; }

    void on_resize(const Fsr2InitParams& params);

private:
    bool initialized_ = false;
    void* scratchBuffer_ = nullptr;
    FfxFsr2Context context_ = {};
    Fsr2InitParams currentParams_ = {};
};

} // namespace gfx
