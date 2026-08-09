#pragma once

#include <glad/glad.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string_view>

namespace gl {

// ── Debug message callback ──

using DebugCallback = void(GLenum source, GLenum type, GLuint id,
                            GLenum severity, GLsizei length,
                            const GLchar* message, const void* user_param);

/// Enable GL_DEBUG_OUTPUT with a default callback that prints to stderr.
/// Pass `synchronous = true` to also enable GL_DEBUG_OUTPUT_SYNCHRONOUS
/// (ensures the callback is invoked in the same thread as the GL call, useful
/// for breakpoint debugging).
void enable_debug_output(bool synchronous = false);

/// Override the global debug callback. Pass nullptr to restore the default.
void set_debug_callback(DebugCallback* cb);

/// Trigger a debug message manually via glDebugMessageInsert.
void debug_message_insert(GLenum source, GLenum type, GLuint id,
                          GLenum severity, std::string_view message);

// ── Object labels (RenderDoc/NSight-friendly naming) ──

inline void label(GLenum identifier, GLuint name, std::string_view label) {
    glObjectLabel(identifier, name, static_cast<GLsizei>(label.size()),
                  label.data());
}

inline void label_buffer(GLuint buffer, std::string_view l) {
    label(GL_BUFFER, buffer, l);
}
inline void label_texture(GLuint texture, std::string_view l) {
    label(GL_TEXTURE, texture, l);
}
inline void label_program(GLuint program, std::string_view l) {
    label(GL_PROGRAM, program, l);
}
inline void label_shader(GLuint shader, std::string_view l) {
    label(GL_SHADER, shader, l);
}
inline void label_vao(GLuint vao, std::string_view l) {
    label(GL_VERTEX_ARRAY, vao, l);
}
inline void label_framebuffer(GLuint fbo, std::string_view l) {
    label(GL_FRAMEBUFFER, fbo, l);
}
inline void label_renderbuffer(GLuint rbo, std::string_view l) {
    label(GL_RENDERBUFFER, rbo, l);
}
inline void label_sampler(GLuint sampler, std::string_view l) {
    label(GL_SAMPLER, sampler, l);
}
inline void label_query(GLuint query, std::string_view l) {
    label(GL_QUERY, query, l);
}
inline void label_program_pipeline(GLuint pipeline, std::string_view l) {
    label(GL_PROGRAM_PIPELINE, pipeline, l);
}

// ── Scoped error checking ──

/// Returns the string name of the current GL error, clearing the error flag.
/// Returns nullptr if no error.
const char* check_gl_error() noexcept;

} // namespace gl

/// Development-time macro: checks GL error after expression and prints to stderr.
/// No-op if NDEBUG is defined.
#ifndef NDEBUG
#define GL_CHECK(expr)                                                     \
    do {                                                                   \
        expr;                                                              \
        const char* __gl_err = gl::check_gl_error();                       \
        if (__gl_err) {                                                    \
            std::fprintf(stderr, "GL error at %s:%d: %s  (after `%s`)\n", \
                         __FILE__, __LINE__, __gl_err, #expr);             \
            std::fflush(stderr);                                           \
        }                                                                  \
    } while (0)
#else
#define GL_CHECK(expr) expr
#endif
