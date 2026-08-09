#include "shader_program.hpp"

#include <gllib/log.hpp>

namespace gl {

ShaderProgram::ShaderProgram(GLenum type, std::string_view source)
    : type_(type)
{
    const char* src = source.data();
    handle_ = glCreateShaderProgramv(type, 1, &src);
}

ShaderProgram::~ShaderProgram() {
    if (handle_) {
        glDeleteProgram(handle_);
    }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
{
    other.handle_ = 0;
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteProgram(handle_);
        }
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = 0;
    }
    return *this;
}

bool ShaderProgram::linked() const {
    GLint status;
    glGetProgramiv(handle_, GL_LINK_STATUS, &status);
    return status == GL_TRUE;
}

std::string ShaderProgram::info_log() const {
    GLint len;
    glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &len);
    if (len == 0) return {};
    std::vector<char> buf(len);
    glGetProgramInfoLog(handle_, len, nullptr, buf.data());
    return std::string(buf.data(), len);
}

GLint ShaderProgram::uniform_location(std::string_view name) const {
    return glGetUniformLocation(handle_, name.data());
}

void ShaderProgram::uniform1i(GLint location, GLint v) {
    glProgramUniform1i(handle_, location, v);
}

void ShaderProgram::uniform1f(GLint location, GLfloat v) {
    glProgramUniform1f(handle_, location, v);
}

void ShaderProgram::uniform2f(GLint location, GLfloat v0, GLfloat v1) {
    glProgramUniform2f(handle_, location, v0, v1);
}

void ShaderProgram::uniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    glProgramUniform3f(handle_, location, v0, v1, v2);
}

void ShaderProgram::uniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    glProgramUniform4f(handle_, location, v0, v1, v2, v3);
}

void ShaderProgram::uniform2fv(GLint location, const GLfloat v[2]) {
    glProgramUniform2fv(handle_, location, 1, v);
}

void ShaderProgram::uniform3fv(GLint location, const GLfloat v[3]) {
    glProgramUniform3fv(handle_, location, 1, v);
}

void ShaderProgram::uniform4fv(GLint location, const GLfloat v[4]) {
    glProgramUniform4fv(handle_, location, 1, v);
}

void ShaderProgram::uniform2iv(GLint location, const GLint v[2]) {
    glProgramUniform2iv(handle_, location, 1, v);
}

void ShaderProgram::uniform3iv(GLint location, const GLint v[3]) {
    glProgramUniform3iv(handle_, location, 1, v);
}

void ShaderProgram::uniform4iv(GLint location, const GLint v[4]) {
    glProgramUniform4iv(handle_, location, 1, v);
}

void ShaderProgram::uniform_matrix3fv(GLint location, const GLfloat* m, bool transpose) {
    glProgramUniformMatrix3fv(handle_, location, 1, transpose ? GL_TRUE : GL_FALSE, m);
}

void ShaderProgram::uniform_matrix4fv(GLint location, const GLfloat* m, bool transpose) {
    glProgramUniformMatrix4fv(handle_, location, 1, transpose ? GL_TRUE : GL_FALSE, m);
}

} // namespace gl
