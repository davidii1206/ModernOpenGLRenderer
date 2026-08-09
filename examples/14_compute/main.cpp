// Example 14 — gl::dispatch_compute + gl::Sync: compute-shader generated particles rendered as points.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>

static const char* comp_src = R"(
#version 460 core
layout (local_size_x = 256) in;

layout (std430, binding = 0) buffer ParticleBuf {
    vec4  positions[];
};

layout (std430, binding = 1) buffer ColorBuf {
    vec4  colors[];
};

layout (location = 0) uniform float u_time;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float a = float(idx) * 0.137f + u_time;
    float b = float(idx) * 0.231f + u_time * 0.7f;
    float c = float(idx) * 0.073f + u_time * 0.4f;
    positions[idx] = vec4(cos(a) * 2.5f + sin(b) * 0.8f,
                          sin(b) * 2.0f + cos(c) * 0.8f,
                          sin(c) * 1.5f + cos(a) * 0.8f,
                          1.0f);
    colors[idx] = vec4(0.5f + 0.5f * sin(a),
                       0.5f + 0.5f * sin(b + 2.0f),
                       0.5f + 0.5f * sin(c + 4.0f),
                       1.0f);
}
)";

static const char* vert_src = R"(
#version 460 core
layout (std430, binding = 0) readonly buffer ParticleBuf {
    vec4 positions[];
};
layout (std430, binding = 1) readonly buffer ColorBuf {
    vec4 colors[];
};
out vec3 v_color;
void main() {
    vec4 pos = positions[gl_VertexID];
    gl_Position = vec4(pos.xy, pos.z * 0.5f, 1.0f);
    v_color = colors[gl_VertexID].rgb;
    gl_PointSize = 12.0f;
}
)";

static const char* frag_src = R"(
#version 460 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    float d = length(gl_PointCoord - vec2(0.5f));
    if (d > 0.5f) discard;
    float a = smoothstep(0.5f, 0.0f, d);
    frag_color = vec4(v_color, a);
}
)";

static constexpr int NUM_PARTICLES = 1024;

int main() {
    gfx::Window window({"14 Compute", 800, 600});

    // --- Compute shader ---
    gl::Shader cs(gl::ShaderType::compute, comp_src);
    if (!cs.compiled()) return EXIT_FAILURE;

    // --- Vertex + fragment shaders (for rendering points) ---
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;

    gl::Program comp_prog;
    comp_prog.attach(cs);
    if (!comp_prog.link()) return EXIT_FAILURE;

    gl::Program draw_prog;
    draw_prog.attach(vs);
    draw_prog.attach(fs);
    if (!draw_prog.link()) return EXIT_FAILURE;

    // --- SSBOs for particles ---
    gl::Buffer pos_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    pos_buf.data(nullptr, NUM_PARTICLES * sizeof(glm::vec4));

    gl::Buffer col_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    col_buf.data(nullptr, NUM_PARTICLES * sizeof(glm::vec4));

    // --- VAO (empty — we draw without attribs, using gl_VertexID) ---
    gl::VertexArray vao;

    gfx::Renderer renderer;
    renderer.set_clear_color(0.02f, 0.02f, 0.04f, 1.0f);
    gl::enable(GL_BLEND);
    gl::blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl::enable(GL_PROGRAM_POINT_SIZE);

    float last_time = window.time();
    while (!window.should_close()) {
        float now = window.time();
        float dt = now - last_time;
        last_time = now;

        // --- Compute pass ---
        comp_prog.use();
        GLint loc = comp_prog.uniform_location("u_time");
        if (loc >= 0) comp_prog.uniform1f(loc, now);
        pos_buf.bind_base(0);
        col_buf.bind_base(1);
        gl::dispatch_compute(NUM_PARTICLES / 256, 1, 1);

        // --- Sync: wait for compute to finish ---
        gl::Sync fence;
        fence.client_wait(1000000000);  // 1 second timeout

        // --- Draw pass ---
        renderer.clear(GL_COLOR_BUFFER_BIT);
        draw_prog.use();
        pos_buf.bind_base(0);
        col_buf.bind_base(1);
        vao.bind();
        gl::draw_arrays(GL_POINTS, 0, NUM_PARTICLES);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
