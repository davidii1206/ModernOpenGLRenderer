#include "program.hpp"
#include "shader.hpp"

#include <gllib/log.hpp>

#include <vector>

namespace gl {

Program::Program() {
    handle_ = glCreateProgram();
}

Program::~Program() {
    if (handle_) {
        glDeleteProgram(handle_);
    }
}

Program::Program(Program&& other) noexcept
    : handle_(other.handle_)
    , linked_(other.linked_)
{
    other.handle_ = 0;
    other.linked_ = false;
}

Program& Program::operator=(Program&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteProgram(handle_);
        }
        handle_ = other.handle_;
        linked_ = other.linked_;
        other.handle_ = 0;
        other.linked_ = false;
    }
    return *this;
}

void Program::attach(const Shader& shader) {
    glAttachShader(handle_, shader.handle());
}

bool Program::link() {
    glLinkProgram(handle_);
    GLint status;
    glGetProgramiv(handle_, GL_LINK_STATUS, &status);
    linked_ = (status == GL_TRUE);
    if (!linked_) {
        gllib::log(gllib::LogLevel::error, info_log().c_str());
    }
    return linked_;
}

void Program::use() const {
    glUseProgram(handle_);
}

void Program::unuse() {
    glUseProgram(0);
}

GLint Program::uniform_location(std::string_view name) const {
    return glGetUniformLocation(handle_, name.data());
}

void Program::uniform1i(GLint location, GLint v) {
    glProgramUniform1i(handle_, location, v);
}

void Program::uniform1ui(GLint location, GLuint v) {
    glProgramUniform1ui(handle_, location, v);
}

void Program::uniform1f(GLint location, GLfloat v) {
    glProgramUniform1f(handle_, location, v);
}

void Program::uniform2f(GLint location, GLfloat v0, GLfloat v1) {
    glProgramUniform2f(handle_, location, v0, v1);
}

void Program::uniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    glProgramUniform3f(handle_, location, v0, v1, v2);
}

void Program::uniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    glProgramUniform4f(handle_, location, v0, v1, v2, v3);
}

void Program::uniform2fv(GLint location, const GLfloat v[2]) {
    glProgramUniform2fv(handle_, location, 1, v);
}

void Program::uniform3fv(GLint location, const GLfloat v[3]) {
    glProgramUniform3fv(handle_, location, 1, v);
}

void Program::uniform4fv(GLint location, const GLfloat v[4]) {
    glProgramUniform4fv(handle_, location, 1, v);
}

void Program::uniform2iv(GLint location, const GLint v[2]) {
    glProgramUniform2iv(handle_, location, 1, v);
}

void Program::uniform3iv(GLint location, const GLint v[3]) {
    glProgramUniform3iv(handle_, location, 1, v);
}

void Program::uniform4iv(GLint location, const GLint v[4]) {
    glProgramUniform4iv(handle_, location, 1, v);
}

void Program::uniform_matrix3fv(GLint location, const GLfloat* m, bool transpose) {
    glProgramUniformMatrix3fv(handle_, location, 1, transpose ? GL_TRUE : GL_FALSE, m);
}

void Program::uniform_matrix4fv(GLint location, const GLfloat* m, bool transpose) {
    glProgramUniformMatrix4fv(handle_, location, 1, transpose ? GL_TRUE : GL_FALSE, m);
}

bool Program::linked() const {
    return linked_;
}

std::string Program::info_log() const {
    GLint len;
    glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &len);
    if (len == 0) return {};
    std::vector<char> buf(len);
    glGetProgramInfoLog(handle_, len, nullptr, buf.data());
    return std::string(buf.data(), len);
}

} // namespace gl
