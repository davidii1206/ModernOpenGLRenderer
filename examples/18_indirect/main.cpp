// Example 18 — gl::multi_draw_elements_indirect: multi-draw indirect with gl_DrawID.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;

layout (std430, binding = 0) readonly buffer OffsetBuf {
    vec2 offsets[];
};

out vec2 v_uv;
flat out int v_id;

void main() {
    vec2 pos = a_pos + offsets[gl_DrawID];
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = a_uv;
    v_id = gl_DrawID;
}
)";

static const char* frag_src = R"(
#version 460 core
in vec2 v_uv;
flat in int v_id;
out vec4 frag_color;
void main() {
    vec3 colors[4] = vec3[](
        vec3(1.0, 0.3, 0.3),
        vec3(0.3, 1.0, 0.3),
        vec3(0.3, 0.3, 1.0),
        vec3(1.0, 1.0, 0.3)
    );
    frag_color = vec4(colors[v_id % 4] * (0.8 + 0.2 * v_uv.x), 1.0);
}
)";

int main() {
    gfx::Window window({"18 Indirect", 800, 600});

    // Single quad (4 verts, 6 indices)
    struct Vertex { float pos[2]; float uv[2]; };
    const Vertex verts[] = {
        {{-0.3f, -0.3f}, {0.0f, 0.0f}},
        {{ 0.3f, -0.3f}, {1.0f, 0.0f}},
        {{ 0.3f,  0.3f}, {1.0f, 1.0f}},
        {{-0.3f,  0.3f}, {0.0f, 1.0f}},
    };
    const unsigned int indices[] = {0, 1, 2, 0, 2, 3};

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(verts, sizeof(verts));
    gl::Buffer ebo(gl::BufferType::index, gl::BufferUsage::static_draw);
    ebo.data(indices, sizeof(indices));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    ebo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, sizeof(Vertex), (void*)0);
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 2, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    vao.enable_attrib(1);

    // Indirect draw buffer: 4 draws of the same quad, different offsets via SSBO
    std::vector<gl::DrawElementsIndirectCommand> cmds = {
        {6, 1, 0, 0, 0},
        {6, 1, 0, 0, 0},
        {6, 1, 0, 0, 0},
        {6, 1, 0, 0, 0},
    };

    gl::Buffer indirect_buf(gl::BufferType::draw, gl::BufferUsage::static_draw);
    indirect_buf.data(cmds.data(), cmds.size() * sizeof(gl::DrawElementsIndirectCommand));

    // SSBO with per-draw offsets
    struct glm2 { float x, y; };
    const glm2 offsets[] = {{-0.35f, 0.35f}, {0.35f, 0.35f}, {-0.35f, -0.35f}, {0.35f, -0.35f}};
    gl::Buffer offset_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    offset_buf.data(offsets, sizeof(offsets));

    // Shader
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    gl::Program prog;
    prog.attach(vs);
    prog.attach(fs);
    if (!prog.link()) return EXIT_FAILURE;

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);

    while (!window.should_close()) {
        renderer.clear(GL_COLOR_BUFFER_BIT);

        prog.use();
        vao.bind();
        offset_buf.bind_base(0);
        indirect_buf.bind();
        gl::multi_draw_elements_indirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 4, 0);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
