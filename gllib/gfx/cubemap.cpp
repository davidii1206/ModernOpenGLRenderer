#include "cubemap.hpp"

#include <gllib/log.hpp>
#include <glm/glm.hpp>
#include <stb_image.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace gfx {

Cubemap::~Cubemap() {
    if (handle_) {
        glDeleteTextures(1, &handle_);
    }
}

Cubemap::Cubemap(Cubemap&& other) noexcept
    : handle_(other.handle_)
    , width_(other.width_)
    , height_(other.height_)
{
    other.handle_ = 0;
    other.width_ = 0;
    other.height_ = 0;
}

Cubemap& Cubemap::operator=(Cubemap&& other) noexcept {
    if (this != &other) {
        if (handle_) glDeleteTextures(1, &handle_);
        handle_ = other.handle_;
        width_ = other.width_;
        height_ = other.height_;
        other.handle_ = 0;
        other.width_ = 0;
        other.height_ = 0;
    }
    return *this;
}

void Cubemap::upload_face(GLenum target, int w, int h, const unsigned char* data) {
    glTexImage2D(target, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}

void Cubemap::upload_face_float(GLenum target, int w, int h, const float* data) {
    glTexImage2D(target, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, data);
}

bool Cubemap::load_from_files(const std::string& pos_x, const std::string& neg_x,
                               const std::string& pos_y, const std::string& neg_y,
                               const std::string& pos_z, const std::string& neg_z)
{
    const std::string files[6] = {pos_x, neg_x, pos_y, neg_y, pos_z, neg_z};
    const GLenum targets[6] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    };

    if (handle_) glDeleteTextures(1, &handle_);
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &handle_);

    // First face determines size; allocate storage after loading it
    int face_w = 0, face_h = 0;

    for (int i = 0; i < 6; ++i) {
        int w, h, channels;
        unsigned char* data = stbi_load(files[i].c_str(), &w, &h, &channels, 4);
        if (!data) {
            gllib::log(gllib::LogLevel::error, "Failed to load cubemap face");
            if (handle_) glDeleteTextures(1, &handle_);
            handle_ = 0;
            return false;
        }

        if (i == 0) {
            face_w = w;
            face_h = h;
            glTextureStorage2D(handle_, 1, GL_SRGB8_ALPHA8, w, h);
        }

        glTextureSubImage3D(handle_, 0, 0, 0, i, w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    }

    width_ = face_w;
    height_ = face_h;

    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return true;
}

void Cubemap::generate_procedural(int size) {
    if (handle_) glDeleteTextures(1, &handle_);
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &handle_);
    width_ = size;
    height_ = size;

    // Face directions: +X, -X, +Y, -Y, +Z, -Z
    const glm::vec3 face_dirs[6] = {
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1},
    };
    const glm::vec3 up_dirs[6] = {
        {0, -1,  0}, {0, -1,  0},
        {0,  0,  1}, {0,  0, -1},
        {0, -1,  0}, {0, -1,  0},
    };

    glTextureStorage2D(handle_, 1, GL_RGB8, size, size);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    std::vector<unsigned char> pixels(static_cast<std::size_t>(size * size * 3));

    for (int face = 0; face < 6; ++face) {
        glm::vec3 f = face_dirs[face];
        glm::vec3 u = up_dirs[face];
        glm::vec3 r = glm::cross(u, f);

        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float px = (2.0f * (x + 0.5f) / size - 1.0f);
                float py = (2.0f * (y + 0.5f) / size - 1.0f);
                glm::vec3 dir = glm::normalize(f + r * px + u * py);

                // Simple gradient sky: blue top, gray-brown bottom
                float t = dir.y * 0.5f + 0.5f;
                glm::vec3 sky_col = glm::mix(
                    glm::vec3(0.6f, 0.5f, 0.4f),  // horizon
                    glm::vec3(0.3f, 0.6f, 1.0f),  // zenith
                    t
                );

                // Sun glow on the +Z face
                if (face == 4) {
                    float sun_dist = glm::length(dir - glm::normalize(glm::vec3(-0.3f, 0.5f, 0.8f)));
                    float sun = std::exp(-sun_dist * sun_dist * 40.0f);
                    sky_col += glm::vec3(1.0f, 0.9f, 0.6f) * sun * 0.8f;
                }

                int idx = (y * size + x) * 3;
                pixels[idx + 0] = static_cast<unsigned char>(glm::clamp(sky_col.r, 0.0f, 1.0f) * 255.0f);
                pixels[idx + 1] = static_cast<unsigned char>(glm::clamp(sky_col.g, 0.0f, 1.0f) * 255.0f);
                pixels[idx + 2] = static_cast<unsigned char>(glm::clamp(sky_col.b, 0.0f, 1.0f) * 255.0f);
            }
        }

        glTextureSubImage3D(handle_, 0, 0, 0, face, size, size, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    }
}

void Cubemap::generate_procedural_hdr(int size, int levels) {
    if (handle_) glDeleteTextures(1, &handle_);
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &handle_);
    width_ = size;
    height_ = size;
    levels_ = levels;

    const glm::vec3 face_dirs[6] = {
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1},
    };
    const glm::vec3 up_dirs[6] = {
        {0, -1,  0}, {0, -1,  0},
        {0,  0,  1}, {0,  0, -1},
        {0, -1,  0}, {0, -1,  0},
    };

    glTextureStorage2D(handle_, levels, GL_RGBA16F, size, size);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    std::vector<float> pixels(static_cast<std::size_t>(size * size * 4));

    for (int face = 0; face < 6; ++face) {
        glm::vec3 f = face_dirs[face];
        glm::vec3 u = up_dirs[face];
        glm::vec3 r = glm::cross(u, f);

        // Sun direction in world space
        glm::vec3 sun_dir = glm::normalize(glm::vec3(-0.3f, 0.5f, 0.8f));

        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float px = (2.0f * (x + 0.5f) / size - 1.0f);
                float py = (2.0f * (y + 0.5f) / size - 1.0f);
                glm::vec3 dir = glm::normalize(f + r * px + u * py);

                // Bright daytime sky
                float t = dir.y * 0.5f + 0.5f;
                glm::vec3 sky_col = glm::mix(
                    glm::vec3(0.7f, 0.8f, 1.0f),      // horizon
                    glm::vec3(0.2f, 0.4f, 1.0f) * 2.0f, // zenith
                    t
                );

                // Sun disk
                float sun_dist = glm::length(dir - sun_dir);
                float sun = std::exp(-sun_dist * sun_dist * 2500.0f);
                sky_col += glm::vec3(1.0f, 0.95f, 0.8f) * sun * 40.0f;

                // Sun glow
                float glow = std::exp(-sun_dist * sun_dist * 60.0f);
                sky_col += glm::vec3(1.0f, 0.8f, 0.4f) * glow * 2.0f;

                int idx = (y * size + x) * 4;
                pixels[idx + 0] = sky_col.r;
                pixels[idx + 1] = sky_col.g;
                pixels[idx + 2] = sky_col.b;
                pixels[idx + 3] = 1.0f;
            }
        }

        glTextureSubImage3D(handle_, 0, 0, 0, face, size, size, 1, GL_RGBA, GL_FLOAT, pixels.data());
    }
}

void Cubemap::create_storage(int size, int levels, GLenum internal_format) {
    if (handle_) glDeleteTextures(1, &handle_);
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &handle_);
    width_ = size;
    height_ = size;
    levels_ = levels;
    glTextureStorage2D(handle_, levels, internal_format, size, size);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, levels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

void Cubemap::bind_image(int face, int level, int unit, GLenum access, GLenum format) const {
    if (handle_) {
        glBindImageTexture(static_cast<GLuint>(unit), handle_, level, GL_FALSE, face, access, format);
    }
}

void Cubemap::bind_image_layered(int level, int unit, GLenum access, GLenum format) const {
    if (handle_) {
        glBindImageTexture(static_cast<GLuint>(unit), handle_, level, GL_TRUE, 0, access, format);
    }
}

void Cubemap::bind(int unit) const {
    if (handle_) {
        glBindTextureUnit(static_cast<GLuint>(unit), handle_);
    }
}

void Cubemap::parameter(GLenum name, GLint value) {
    if (handle_) {
        glTextureParameteri(handle_, name, value);
    }
}

void Cubemap::generate_mipmap() {
    if (handle_) {
        glGenerateTextureMipmap(handle_);
    }
}

} // namespace gfx
