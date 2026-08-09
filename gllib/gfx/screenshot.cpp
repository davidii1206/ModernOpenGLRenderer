#include "screenshot.hpp"

#include <glad/glad.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace gfx {

bool screenshot(const char* filename, int x, int y, int width, int height) {
    // Default to current viewport if dimensions are 0
    if (width == 0 || height == 0) {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        x = viewport[0];
        y = viewport[1];
        width = viewport[2];
        height = viewport[3];
    }

    std::vector<unsigned char> pixels(width * height * 4);
    glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // stb_image_write expects top-down rows; OpenGL gives bottom-up.
    // Flip the image in-place.
    std::vector<unsigned char> row(width * 4);
    for (int i = 0; i < height / 2; ++i) {
        unsigned char* top = pixels.data() + i * width * 4;
        unsigned char* bot = pixels.data() + (height - 1 - i) * width * 4;
        std::memcpy(row.data(), top, width * 4);
        std::memcpy(top, bot, width * 4);
        std::memcpy(bot, row.data(), width * 4);
    }

    int ret = stbi_write_png(filename, width, height, 4, pixels.data(), width * 4);
    if (!ret) {
        std::fprintf(stderr, "screenshot: failed to write %s\n", filename);
        return false;
    }
    return true;
}

} // namespace gfx
