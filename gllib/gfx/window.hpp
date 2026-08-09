#pragma once

namespace gfx {

enum class Key {
    unknown = -1,
    space, apostrophe, comma, minus, period, slash,
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9,
    semicolon, equal,
    a, b, c, d, e, f, g, h, i, j, k, l, m,
    n, o, p, q, r, s, t, u, v, w, x, y, z,
    left_bracket, backslash, right_bracket,
    escape, enter, tab, backspace, insert, del,
    right, left, down, up,
    shift, control, alt, super,
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
};

enum class MouseButton {
    left = 0,
    right = 1,
    middle = 2,
};

struct WindowDesc {
    const char* title = "gllib";
    int width = 800;
    int height = 600;
    bool vsync = true;
    bool resizable = true;
    bool debug = false; // request a debug OpenGL context (for glDebugMessageCallback)
    int gl_major = 4;
    int gl_minor = 6;
};

class Window {
public:
    explicit Window(const WindowDesc& desc = {});
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    bool should_close() const;
    void poll_events();
    void swap_buffers();
    void vsync(bool enabled);
    bool vsync() const;

    int width() const;
    int height() const;
    int framebuffer_width() const;
    int framebuffer_height() const;
    float time() const;

    bool key_down(Key key) const;
    bool mouse_down(MouseButton button) const;
    void cursor_position(double& x, double& y) const;
    double scroll_delta();
    void* native_handle() const;

private:
    void* impl_ = nullptr;
};

} // namespace gfx
