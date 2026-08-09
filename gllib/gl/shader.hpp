#pragma once

#include <glad/glad.h>
#include <string>
#include <string_view>

namespace gl {

enum class ShaderType : GLenum {
    vertex   = GL_VERTEX_SHADER,
    fragment = GL_FRAGMENT_SHADER,
    geometry = GL_GEOMETRY_SHADER,
    compute  = GL_COMPUTE_SHADER,
    tess_control = GL_TESS_CONTROL_SHADER,
    tess_eval    = GL_TESS_EVALUATION_SHADER,
};

class Shader {
public:
    Shader() = default;
    Shader(ShaderType type, std::string_view source);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;

    bool valid() const { return handle_ != 0; }
    bool compiled() const;
    std::string info_log() const;

    GLuint handle() const { return handle_; }
    ShaderType type() const { return type_; }

    // SPIR-V: create from binary SPIR-V data
    static Shader from_spirv(ShaderType type, const void* data, size_t size, const char* entry_point = "main");

private:
    GLuint handle_ = 0;
    ShaderType type_ = ShaderType::vertex;
};

} // namespace gl
