#include "window.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <gl/state.hpp>

namespace gfx {
namespace {

struct WindowData {
    GLFWwindow* handle = nullptr;
    int width = 0;
    int height = 0;
    int fb_width = 0;
    int fb_height = 0;
    bool vsync = true;
    double scroll_y = 0.0;
};

Key to_gfx_key(int glfw_key) {
    switch (glfw_key) {
        case GLFW_KEY_SPACE:       return Key::space;
        case GLFW_KEY_APOSTROPHE:  return Key::apostrophe;
        case GLFW_KEY_COMMA:       return Key::comma;
        case GLFW_KEY_MINUS:       return Key::minus;
        case GLFW_KEY_PERIOD:      return Key::period;
        case GLFW_KEY_SLASH:       return Key::slash;
        case GLFW_KEY_0:           return Key::_0;
        case GLFW_KEY_1:           return Key::_1;
        case GLFW_KEY_2:           return Key::_2;
        case GLFW_KEY_3:           return Key::_3;
        case GLFW_KEY_4:           return Key::_4;
        case GLFW_KEY_5:           return Key::_5;
        case GLFW_KEY_6:           return Key::_6;
        case GLFW_KEY_7:           return Key::_7;
        case GLFW_KEY_8:           return Key::_8;
        case GLFW_KEY_9:           return Key::_9;
        case GLFW_KEY_SEMICOLON:   return Key::semicolon;
        case GLFW_KEY_EQUAL:       return Key::equal;
        case GLFW_KEY_A:           return Key::a;
        case GLFW_KEY_B:           return Key::b;
        case GLFW_KEY_C:           return Key::c;
        case GLFW_KEY_D:           return Key::d;
        case GLFW_KEY_E:           return Key::e;
        case GLFW_KEY_F:           return Key::f;
        case GLFW_KEY_G:           return Key::g;
        case GLFW_KEY_H:           return Key::h;
        case GLFW_KEY_I:           return Key::i;
        case GLFW_KEY_J:           return Key::j;
        case GLFW_KEY_K:           return Key::k;
        case GLFW_KEY_L:           return Key::l;
        case GLFW_KEY_M:           return Key::m;
        case GLFW_KEY_N:           return Key::n;
        case GLFW_KEY_O:           return Key::o;
        case GLFW_KEY_P:           return Key::p;
        case GLFW_KEY_Q:           return Key::q;
        case GLFW_KEY_R:           return Key::r;
        case GLFW_KEY_S:           return Key::s;
        case GLFW_KEY_T:           return Key::t;
        case GLFW_KEY_U:           return Key::u;
        case GLFW_KEY_V:           return Key::v;
        case GLFW_KEY_W:           return Key::w;
        case GLFW_KEY_X:           return Key::x;
        case GLFW_KEY_Y:           return Key::y;
        case GLFW_KEY_Z:           return Key::z;
        case GLFW_KEY_LEFT_BRACKET:  return Key::left_bracket;
        case GLFW_KEY_BACKSLASH:     return Key::backslash;
        case GLFW_KEY_RIGHT_BRACKET: return Key::right_bracket;
        case GLFW_KEY_ESCAPE:     return Key::escape;
        case GLFW_KEY_ENTER:      return Key::enter;
        case GLFW_KEY_TAB:        return Key::tab;
        case GLFW_KEY_BACKSPACE:  return Key::backspace;
        case GLFW_KEY_INSERT:     return Key::insert;
        case GLFW_KEY_DELETE:     return Key::del;
        case GLFW_KEY_RIGHT:      return Key::right;
        case GLFW_KEY_LEFT:       return Key::left;
        case GLFW_KEY_DOWN:       return Key::down;
        case GLFW_KEY_UP:         return Key::up;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:   return Key::shift;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL: return Key::control;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:     return Key::alt;
        case GLFW_KEY_LEFT_SUPER:
        case GLFW_KEY_RIGHT_SUPER:   return Key::super;
        case GLFW_KEY_F1:  return Key::f1;
        case GLFW_KEY_F2:  return Key::f2;
        case GLFW_KEY_F3:  return Key::f3;
        case GLFW_KEY_F4:  return Key::f4;
        case GLFW_KEY_F5:  return Key::f5;
        case GLFW_KEY_F6:  return Key::f6;
        case GLFW_KEY_F7:  return Key::f7;
        case GLFW_KEY_F8:  return Key::f8;
        case GLFW_KEY_F9:  return Key::f9;
        case GLFW_KEY_F10: return Key::f10;
        case GLFW_KEY_F11: return Key::f11;
        case GLFW_KEY_F12: return Key::f12;
        default:           return Key::unknown;
    }
}

int to_glfw_key(Key key) {
    switch (key) {
        case Key::space:        return GLFW_KEY_SPACE;
        case Key::apostrophe:   return GLFW_KEY_APOSTROPHE;
        case Key::comma:        return GLFW_KEY_COMMA;
        case Key::minus:        return GLFW_KEY_MINUS;
        case Key::period:       return GLFW_KEY_PERIOD;
        case Key::slash:        return GLFW_KEY_SLASH;
        case Key::_0:           return GLFW_KEY_0;
        case Key::_1:           return GLFW_KEY_1;
        case Key::_2:           return GLFW_KEY_2;
        case Key::_3:           return GLFW_KEY_3;
        case Key::_4:           return GLFW_KEY_4;
        case Key::_5:           return GLFW_KEY_5;
        case Key::_6:           return GLFW_KEY_6;
        case Key::_7:           return GLFW_KEY_7;
        case Key::_8:           return GLFW_KEY_8;
        case Key::_9:           return GLFW_KEY_9;
        case Key::semicolon:    return GLFW_KEY_SEMICOLON;
        case Key::equal:        return GLFW_KEY_EQUAL;
        case Key::a:            return GLFW_KEY_A;
        case Key::b:            return GLFW_KEY_B;
        case Key::c:            return GLFW_KEY_C;
        case Key::d:            return GLFW_KEY_D;
        case Key::e:            return GLFW_KEY_E;
        case Key::f:            return GLFW_KEY_F;
        case Key::g:            return GLFW_KEY_G;
        case Key::h:            return GLFW_KEY_H;
        case Key::i:            return GLFW_KEY_I;
        case Key::j:            return GLFW_KEY_J;
        case Key::k:            return GLFW_KEY_K;
        case Key::l:            return GLFW_KEY_L;
        case Key::m:            return GLFW_KEY_M;
        case Key::n:            return GLFW_KEY_N;
        case Key::o:            return GLFW_KEY_O;
        case Key::p:            return GLFW_KEY_P;
        case Key::q:            return GLFW_KEY_Q;
        case Key::r:            return GLFW_KEY_R;
        case Key::s:            return GLFW_KEY_S;
        case Key::t:            return GLFW_KEY_T;
        case Key::u:            return GLFW_KEY_U;
        case Key::v:            return GLFW_KEY_V;
        case Key::w:            return GLFW_KEY_W;
        case Key::x:            return GLFW_KEY_X;
        case Key::y:            return GLFW_KEY_Y;
        case Key::z:            return GLFW_KEY_Z;
        case Key::left_bracket: return GLFW_KEY_LEFT_BRACKET;
        case Key::backslash:    return GLFW_KEY_BACKSLASH;
        case Key::right_bracket:return GLFW_KEY_RIGHT_BRACKET;
        case Key::escape:       return GLFW_KEY_ESCAPE;
        case Key::enter:        return GLFW_KEY_ENTER;
        case Key::tab:          return GLFW_KEY_TAB;
        case Key::backspace:    return GLFW_KEY_BACKSPACE;
        case Key::insert:       return GLFW_KEY_INSERT;
        case Key::del:          return GLFW_KEY_DELETE;
        case Key::right:        return GLFW_KEY_RIGHT;
        case Key::left:         return GLFW_KEY_LEFT;
        case Key::down:         return GLFW_KEY_DOWN;
        case Key::up:           return GLFW_KEY_UP;
        case Key::shift:        return GLFW_KEY_LEFT_SHIFT;
        case Key::control:      return GLFW_KEY_LEFT_CONTROL;
        case Key::alt:          return GLFW_KEY_LEFT_ALT;
        case Key::super:        return GLFW_KEY_LEFT_SUPER;
        case Key::f1:           return GLFW_KEY_F1;
        case Key::f2:           return GLFW_KEY_F2;
        case Key::f3:           return GLFW_KEY_F3;
        case Key::f4:           return GLFW_KEY_F4;
        case Key::f5:           return GLFW_KEY_F5;
        case Key::f6:           return GLFW_KEY_F6;
        case Key::f7:           return GLFW_KEY_F7;
        case Key::f8:           return GLFW_KEY_F8;
        case Key::f9:           return GLFW_KEY_F9;
        case Key::f10:          return GLFW_KEY_F10;
        case Key::f11:          return GLFW_KEY_F11;
        case Key::f12:          return GLFW_KEY_F12;
        default:                return GLFW_KEY_UNKNOWN;
    }
}

bool glfw_initialized = false;

WindowData* to_data(void* ptr) {
    return static_cast<WindowData*>(ptr);
}

void framebuffer_resize_callback(GLFWwindow* handle, int w, int h) {
    auto* d = static_cast<WindowData*>(glfwGetWindowUserPointer(handle));
    if (d) {
        d->fb_width = w;
        d->fb_height = h;
        gl::viewport(0, 0, w, h);
    }
}

void scroll_callback(GLFWwindow* handle, double xoffset, double yoffset) {
    auto* d = static_cast<WindowData*>(glfwGetWindowUserPointer(handle));
    if (d) {
        d->scroll_y += yoffset;
    }
}

} // namespace

Window::Window(const WindowDesc& desc) {
    auto* d = new WindowData();
    d->width = desc.width;
    d->height = desc.height;

    if (!glfw_initialized) {
        glfwInit();
        glfw_initialized = true;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, desc.gl_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, desc.gl_minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,
                   desc.debug ? GLFW_TRUE : GLFW_FALSE);

    d->handle = glfwCreateWindow(desc.width, desc.height, desc.title, nullptr, nullptr);

    glfwMakeContextCurrent(d->handle);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glfwSetWindowUserPointer(d->handle, d);

    glfwGetFramebufferSize(d->handle, &d->fb_width, &d->fb_height);
    glfwSetFramebufferSizeCallback(d->handle, framebuffer_resize_callback);
    glfwSetScrollCallback(d->handle, scroll_callback);
    gl::viewport(0, 0, d->fb_width, d->fb_height);

    d->vsync = desc.vsync;
    glfwSwapInterval(d->vsync ? 1 : 0);

    impl_ = d;
}

Window::~Window() {
    if (auto* d = to_data(impl_)) {
        if (d->handle) {
            glfwDestroyWindow(d->handle);
        }
        delete d;
    }
}

Window::Window(Window&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

Window& Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        if (auto* d = to_data(impl_)) {
            if (d->handle) glfwDestroyWindow(d->handle);
            delete d;
        }
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool Window::should_close() const {
    return glfwWindowShouldClose(to_data(impl_)->handle);
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::swap_buffers() {
    glfwSwapBuffers(to_data(impl_)->handle);
}

void Window::vsync(bool enabled) {
    to_data(impl_)->vsync = enabled;
    glfwSwapInterval(enabled ? 1 : 0);
}

bool Window::vsync() const {
    return to_data(impl_)->vsync;
}

int Window::width() const {
    auto* d = to_data(impl_);
    glfwGetWindowSize(d->handle, &d->width, &d->height);
    return d->width;
}

int Window::height() const {
    auto* d = to_data(impl_);
    glfwGetWindowSize(d->handle, &d->width, &d->height);
    return d->height;
}

int Window::framebuffer_width() const {
    return to_data(impl_)->fb_width;
}

int Window::framebuffer_height() const {
    return to_data(impl_)->fb_height;
}

float Window::time() const {
    return static_cast<float>(glfwGetTime());
}

bool Window::key_down(Key key) const {
    return glfwGetKey(to_data(impl_)->handle, to_glfw_key(key)) == GLFW_PRESS;
}

bool Window::mouse_down(MouseButton button) const {
    return glfwGetMouseButton(to_data(impl_)->handle, static_cast<int>(button)) == GLFW_PRESS;
}

void Window::cursor_position(double& x, double& y) const {
    glfwGetCursorPos(to_data(impl_)->handle, &x, &y);
}

void* Window::native_handle() const {
    return static_cast<void*>(to_data(impl_)->handle);
}

double Window::scroll_delta() {
    auto* d = to_data(impl_);
    double delta = d->scroll_y;
    d->scroll_y = 0.0;
    return delta;
}

} // namespace gfx
