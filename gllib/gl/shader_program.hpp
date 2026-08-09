#pragma once

#include <glad/glad.h>
#include <string>
#include <string_view>
#include <vector>

namespace gl {

class ShaderProgram {
public:
    ShaderProgram(GLenum type, std::string_view source);
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    [[nodiscard]] GLint uniform_location(std::string_view name) const;

    void uniform1i(GLint location, GLint v);
    void uniform1f(GLint location, GLfloat v);
    void uniform2f(GLint location, GLfloat v0, GLfloat v1);
    void uniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
    void uniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    void uniform2fv(GLint location, const GLfloat v[2]);
    void uniform3fv(GLint location, const GLfloat v[3]);
    void uniform4fv(GLint location, const GLfloat v[4]);
    void uniform2iv(GLint location, const GLint v[2]);
    void uniform3iv(GLint location, const GLint v[3]);
    void uniform4iv(GLint location, const GLint v[4]);
    void uniform_matrix3fv(GLint location, const GLfloat* m, bool transpose = false);
    void uniform_matrix4fv(GLint location, const GLfloat* m, bool transpose = false);

    bool valid() const { return handle_ != 0; }
    bool linked() const;
    std::string info_log() const;

    GLuint handle() const { return handle_; }
    GLenum type() const { return type_; }

private:
    GLuint handle_ = 0;
    GLenum type_ = GL_VERTEX_SHADER;
};

} // namespace gl
