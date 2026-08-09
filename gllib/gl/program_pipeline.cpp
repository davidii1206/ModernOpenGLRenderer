#include "program_pipeline.hpp"
#include "shader_program.hpp"

#include <gllib/log.hpp>

#include <vector>

namespace gl {

ProgramPipeline::ProgramPipeline() {
    glCreateProgramPipelines(1, &handle_);
}

ProgramPipeline::~ProgramPipeline() {
    if (handle_) {
        glDeleteProgramPipelines(1, &handle_);
    }
}

ProgramPipeline::ProgramPipeline(ProgramPipeline&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = 0;
}

ProgramPipeline& ProgramPipeline::operator=(ProgramPipeline&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteProgramPipelines(1, &handle_);
        }
        handle_ = other.handle_;
        other.handle_ = 0;
    }
    return *this;
}

void ProgramPipeline::use_stages(GLbitfield stages, const ShaderProgram& program) {
    glUseProgramStages(handle_, stages, program.handle());
}

bool ProgramPipeline::validate() {
    glValidateProgramPipeline(handle_);
    GLint status;
    glGetProgramPipelineiv(handle_, GL_VALIDATE_STATUS, &status);
    if (status != GL_TRUE) {
        gllib::log(gllib::LogLevel::error, info_log().c_str());
    }
    return status == GL_TRUE;
}

void ProgramPipeline::bind() {
    glBindProgramPipeline(handle_);
}

void ProgramPipeline::unbind() {
    glBindProgramPipeline(0);
}

std::string ProgramPipeline::info_log() const {
    GLint len;
    glGetProgramPipelineiv(handle_, GL_INFO_LOG_LENGTH, &len);
    if (len == 0) return {};
    std::vector<char> buf(len);
    glGetProgramPipelineInfoLog(handle_, len, nullptr, buf.data());
    return std::string(buf.data(), len);
}

} // namespace gl
