#pragma once

#include <glad/glad.h>

namespace gl {
class Texture;
class Program;
} // namespace gl

namespace gfx {

class Cubemap;

class IBLProbe {
public:
    IBLProbe();
    ~IBLProbe();

    IBLProbe(const IBLProbe&) = delete;
    IBLProbe& operator=(const IBLProbe&) = delete;

    IBLProbe(IBLProbe&&) noexcept;
    IBLProbe& operator=(IBLProbe&&) noexcept;

    void generate_procedural(int size = 256);
    void bake();

    const Cubemap& env_map() const { return *env_map_; }
    const Cubemap& irradiance_map() const { return *irradiance_map_; }
    const Cubemap& prefilter_map() const { return *prefilter_map_; }
    const gl::Texture& brdf_lut() const { return *brdf_lut_; }

private:
    Cubemap* env_map_ = nullptr;
    Cubemap* irradiance_map_ = nullptr;
    Cubemap* prefilter_map_ = nullptr;
    gl::Texture* brdf_lut_ = nullptr;
};

} // namespace gfx
