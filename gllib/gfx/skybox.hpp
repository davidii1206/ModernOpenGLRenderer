#pragma once

#include <glad/glad.h>

namespace gl {
class Program;
class VertexArray;
class Buffer;
} // namespace gl

namespace gfx {

class Cubemap;
class Camera;

class Skybox {
public:
    explicit Skybox(const Cubemap& cubemap);
    ~Skybox();

    Skybox(const Skybox&) = delete;
    Skybox& operator=(const Skybox&) = delete;

    void render(const Camera& camera);

private:
    const Cubemap* cubemap_;
    gl::Program* program_ = nullptr;
    gl::VertexArray* vao_ = nullptr;
    gl::Buffer* vbo_ = nullptr;
    gl::Buffer* ebo_ = nullptr;
};

} // namespace gfx
