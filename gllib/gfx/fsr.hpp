#pragma once

#include <cstdint>
#include <cmath>

namespace gfx {

inline void fsr_easu_con(
    uint32_t con[4],
    float inputViewportInPixelsX, float inputViewportInPixelsY,
    float inputSizeInPixelsX, float inputSizeInPixelsY,
    float outputSizeInPixelsX, float outputSizeInPixelsY)
{
    con[0] = 0;
    auto rcp = [](float x) -> float { return 1.0f / x; };

    // Output pixel position to viewport pixel position
    reinterpret_cast<float&>(con[0]) = inputViewportInPixelsX * rcp(outputSizeInPixelsX);
    reinterpret_cast<float&>(con[1]) = inputViewportInPixelsY * rcp(outputSizeInPixelsY);
    reinterpret_cast<float&>(con[2]) = 0.5f * inputViewportInPixelsX * rcp(outputSizeInPixelsX) - 0.5f;
    reinterpret_cast<float&>(con[3]) = 0.5f * inputViewportInPixelsY * rcp(outputSizeInPixelsY) - 0.5f;
}

inline void fsr_easu_con1(
    uint32_t con1[4],
    float inputSizeInPixelsX, float inputSizeInPixelsY)
{
    auto rcp = [](float x) -> float { return 1.0f / x; };
    reinterpret_cast<float&>(con1[0]) = rcp(inputSizeInPixelsX);
    reinterpret_cast<float&>(con1[1]) = rcp(inputSizeInPixelsY);
    reinterpret_cast<float&>(con1[2]) = 1.0f * rcp(inputSizeInPixelsX);
    reinterpret_cast<float&>(con1[3]) = -1.0f * rcp(inputSizeInPixelsY);
}

inline void fsr_easu_con2(
    uint32_t con2[4],
    float inputSizeInPixelsX, float inputSizeInPixelsY)
{
    auto rcp = [](float x) -> float { return 1.0f / x; };
    reinterpret_cast<float&>(con2[0]) = -1.0f * rcp(inputSizeInPixelsX);
    reinterpret_cast<float&>(con2[1]) = 2.0f * rcp(inputSizeInPixelsY);
    reinterpret_cast<float&>(con2[2]) = 1.0f * rcp(inputSizeInPixelsX);
    reinterpret_cast<float&>(con2[3]) = 2.0f * rcp(inputSizeInPixelsY);
}

inline void fsr_easu_con3(
    uint32_t con3[4],
    float inputSizeInPixelsX, float inputSizeInPixelsY)
{
    auto rcp = [](float x) -> float { return 1.0f / x; };
    reinterpret_cast<float&>(con3[0]) = 0.0f * rcp(inputSizeInPixelsX);
    reinterpret_cast<float&>(con3[1]) = 4.0f * rcp(inputSizeInPixelsY);
    con3[2] = 0;
    con3[3] = 0;
}

inline void fsr_rcas_con(
    uint32_t con[4],
    float sharpness)
{
    sharpness = std::exp2(-sharpness);
    reinterpret_cast<float&>(con[0]) = sharpness;
    reinterpret_cast<float&>(con[1]) = sharpness;
    // Actually: con[1] = AU1_AH2_AF2(hSharp) packs two halfs in one uint
    // For full float path, RCAS only uses con[0]; con[1] is for half path
    con[2] = 0;
    con[3] = 0;
}

} // namespace gfx
