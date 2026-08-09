#pragma once

#include <glad/glad.h>

namespace gl {

struct DrawArraysIndirectCommand {
    GLuint count;
    GLuint instanceCount;
    GLuint first;
    GLuint baseInstance;
};

struct DrawElementsIndirectCommand {
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};

} // namespace gl
