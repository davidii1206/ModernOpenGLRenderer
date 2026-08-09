#pragma once

namespace gfx {

/// Save the current read framebuffer to a PNG file.
/// Uses current GL_READ_BUFFER and GL_VIEWPORT if x/y/width/height are 0.
/// Returns true on success.
bool screenshot(const char* filename,
                int x = 0, int y = 0,
                int width = 0, int height = 0);

} // namespace gfx
