#pragma once

#include <glad/glad.h>
#include <string>

namespace gl {

class ShaderProgram;

class ProgramPipeline {
public:
    ProgramPipeline();
    ~ProgramPipeline();

    ProgramPipeline(const ProgramPipeline&) = delete;
    ProgramPipeline& operator=(const ProgramPipeline&) = delete;

    ProgramPipeline(ProgramPipeline&& other) noexcept;
    ProgramPipeline& operator=(ProgramPipeline&& other) noexcept;

    void use_stages(GLbitfield stages, const ShaderProgram& program);
    bool validate();
    void bind();
    static void unbind();

    bool valid() const { return handle_ != 0; }
    std::string info_log() const;

    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
};

} // namespace gl
