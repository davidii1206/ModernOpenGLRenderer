#pragma once

#include "shader.hpp"
#include "program.hpp"

namespace gl {

Shader shader_from_file(ShaderType type, const char* path);
Program program_from_files(const char* vert_path, const char* frag_path);
Program program_from_files(const char* vert_path, const char* frag_path,
                           const char* geom_path);

} // namespace gl
