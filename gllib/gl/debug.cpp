#include "debug.hpp"

#include <cstdio>
#include <cstring>

namespace gl {

// ── Internal state ──

static DebugCallback* s_user_callback = nullptr;

static const char* source_str(GLenum source) {
    switch (source) {
    case GL_DEBUG_SOURCE_API:             return "API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   return "WINDOW";
    case GL_DEBUG_SOURCE_SHADER_COMPILER: return "SHADER_COMPILER";
    case GL_DEBUG_SOURCE_THIRD_PARTY:     return "THIRD_PARTY";
    case GL_DEBUG_SOURCE_APPLICATION:     return "APPLICATION";
    case GL_DEBUG_SOURCE_OTHER:           return "OTHER";
    default:                              return "?";
    }
}

static const char* type_str(GLenum type) {
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:               return "ERROR";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  return "UNDEFINED_BEHAVIOR";
    case GL_DEBUG_TYPE_PORTABILITY:         return "PORTABILITY";
    case GL_DEBUG_TYPE_PERFORMANCE:         return "PERFORMANCE";
    case GL_DEBUG_TYPE_MARKER:              return "MARKER";
    case GL_DEBUG_TYPE_PUSH_GROUP:          return "PUSH_GROUP";
    case GL_DEBUG_TYPE_POP_GROUP:           return "POP_GROUP";
    case GL_DEBUG_TYPE_OTHER:               return "OTHER";
    default:                                return "?";
    }
}

static const char* severity_str(GLenum severity) {
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:         return "HIGH";
    case GL_DEBUG_SEVERITY_MEDIUM:       return "MEDIUM";
    case GL_DEBUG_SEVERITY_LOW:          return "LOW";
    case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
    default:                             return "?";
    }
}

static void GLAPIENTRY default_callback(GLenum source, GLenum type, GLuint id,
                                         GLenum severity, GLsizei /*length*/,
                                         const GLchar* message,
                                         const void* /*user_param*/)
{
    // Skip notifications by default (too noisy)
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;

    std::fprintf(stderr, "[GL] %s %s %s (id=%u): %s\n",
                 severity_str(severity), source_str(source), type_str(type),
                 id, message);
    std::fflush(stderr);

    // Trap on high-severity errors for debugging (skip shader compiler —
    // those are handled gracefully by the application)
    if (severity == GL_DEBUG_SEVERITY_HIGH && source != GL_DEBUG_SOURCE_SHADER_COMPILER) {
        std::fprintf(stderr, "[GL] High-severity error — aborting\n");
        std::abort();
    }
}

static void GLAPIENTRY dispatch_callback(GLenum source, GLenum type,
                                          GLuint id, GLenum severity,
                                          GLsizei length,
                                          const GLchar* message,
                                          const void* user_param)
{
    if (s_user_callback) {
        s_user_callback(source, type, id, severity, length, message, user_param);
    } else {
        default_callback(source, type, id, severity, length, message, user_param);
    }
}

// ── Public API ──

void enable_debug_output(bool synchronous) {
    glEnable(GL_DEBUG_OUTPUT);
    if (synchronous) {
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }
    glDebugMessageCallback(dispatch_callback, nullptr);
}

void set_debug_callback(DebugCallback* cb) {
    s_user_callback = cb;
}

void debug_message_insert(GLenum source, GLenum type, GLuint id,
                          GLenum severity, std::string_view message)
{
    glDebugMessageInsert(source, type, id, severity,
                         static_cast<GLsizei>(message.size()),
                         message.data());
}

const char* check_gl_error() noexcept {
    switch (glGetError()) {
    case GL_NO_ERROR:      return nullptr;
    case GL_INVALID_ENUM:  return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
    case GL_STACK_OVERFLOW:  return "GL_STACK_OVERFLOW";
    default:               return "GL_UNKNOWN_ERROR";
    }
}

} // namespace gl
