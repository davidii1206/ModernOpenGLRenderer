#include "skybox.hpp"
#include "cubemap.hpp"
#include "camera.hpp"

#include <gl/buffer.hpp>
#include <gl/program.hpp>
#include <gl/shader.hpp>
#include <gl/state.hpp>
#include <gl/vertex_array.hpp>

namespace gfx {

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
uniform mat4 u_view_proj;
out vec3 v_uvw;
void main() {
    v_uvw = a_pos;
    vec4 p = u_view_proj * vec4(a_pos, 1.0);
    gl_Position = p.xyww;
}
)";

static const char* frag_src = R"(
#version 460 core
uniform samplerCube u_skybox;
in vec3 v_uvw;
out vec4 frag_color;

vec3 aces(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_skybox, normalize(v_uvw)).rgb;
    frag_color = vec4(aces(hdr), 1.0);
}
)";

Skybox::Skybox(const Cubemap& cubemap)
    : cubemap_(&cubemap)
{
    float verts[] = {
        -1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
        -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,
    };
    unsigned int idx[] = {
        0, 1, 2, 0, 2, 3,
        5, 4, 7, 5, 7, 6,
        4, 0, 3, 4, 3, 7,
        1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,
        4, 5, 1, 4, 1, 0,
    };

    vbo_ = new gl::Buffer(gl::BufferType::vertex);
    vbo_->data(verts, sizeof(verts));

    ebo_ = new gl::Buffer(gl::BufferType::index);
    ebo_->data(idx, sizeof(idx));

    vao_ = new gl::VertexArray;
    vao_->bind();
    vbo_->bind();
    ebo_->bind();
    vao_->attrib_pointer(0, 3, GL_FLOAT, false, 12, (void*)0);
    vao_->enable_attrib(0);
    gl::VertexArray::unbind();

    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (vs.compiled() && fs.compiled()) {
        program_ = new gl::Program;
        program_->attach(vs);
        program_->attach(fs);
        program_->link();
    }
}

Skybox::~Skybox() {
    delete vbo_;
    delete ebo_;
    delete vao_;
    delete program_;
}

void Skybox::render(const Camera& camera) {
    if (!program_) return;

    bool prev_cull = gl::is_enabled(GL_CULL_FACE);
    GLboolean prev_depth_mask;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &prev_depth_mask);
    GLint prev_depth_func;
    glGetIntegerv(GL_DEPTH_FUNC, &prev_depth_func);

    gl::disable(GL_CULL_FACE);
    gl::depth_mask(GL_FALSE);
    gl::depth_func(GL_LEQUAL);

    program_->use();
    cubemap_->bind(0);

    // Remove translation from view matrix for the skybox
    glm::mat4 view = camera.view();
    view[3][0] = 0.0f;
    view[3][1] = 0.0f;
    view[3][2] = 0.0f;
    glm::mat4 vp = camera.projection() * view;

    GLint loc = program_->uniform_location("u_view_proj");
    if (loc >= 0) program_->uniform_matrix4fv(loc, &vp[0][0]);

    GLint tex_loc = program_->uniform_location("u_skybox");
    if (tex_loc >= 0) program_->uniform1i(tex_loc, 0);

    vao_->bind();
    gl::draw_elements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr);

    // Restore state
    if (prev_cull) gl::enable(GL_CULL_FACE);
    gl::depth_mask(prev_depth_mask);
    gl::depth_func(prev_depth_func);
}

} // namespace gfx
