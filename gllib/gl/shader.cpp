#include "shader.hpp"

#include <gllib/log.hpp>

#include <cstddef>
#include <vector>

namespace gl {

Shader::Shader(ShaderType type, std::string_view source)
    : type_(type)
{
    handle_ = glCreateShader(static_cast<GLenum>(type_));
    const GLchar* src = source.data();
    GLint len = static_cast<GLint>(source.size());
    glShaderSource(handle_, 1, &src, &len);
    glCompileShader(handle_);
    if (!compiled()) {
        gllib::log(gllib::LogLevel::error, info_log().c_str());
    }
}

Shader::~Shader() {
    if (handle_) {
        glDeleteShader(handle_);
    }
}

Shader::Shader(Shader&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
{
    other.handle_ = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteShader(handle_);
        }
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = 0;
    }
    return *this;
}

bool Shader::compiled() const {
    if (!handle_) return false;
    GLint status;
    glGetShaderiv(handle_, GL_COMPILE_STATUS, &status);
    return status == GL_TRUE;
}

std::string Shader::info_log() const {
    if (!handle_) return {};
    GLint len;
    glGetShaderiv(handle_, GL_INFO_LOG_LENGTH, &len);
    if (len == 0) return {};
    std::vector<char> buf(static_cast<std::size_t>(len));
    glGetShaderInfoLog(handle_, len, nullptr, buf.data());
    return std::string(buf.data(), static_cast<std::size_t>(len));
}

Shader Shader::from_spirv(ShaderType type, const void* data, size_t size, const char* entry_point) {
    Shader shader;
    shader.type_ = type;
    shader.handle_ = glCreateShader(static_cast<GLenum>(type));
    glShaderBinary(1, &shader.handle_, GL_SHADER_BINARY_FORMAT_SPIR_V, data, static_cast<GLsizei>(size));
    glSpecializeShader(shader.handle_, entry_point, 0, nullptr, nullptr);
    if (!shader.compiled()) {
        gllib::log(gllib::LogLevel::error, shader.info_log().c_str());
    }
    return shader;
}

} // namespace gl
