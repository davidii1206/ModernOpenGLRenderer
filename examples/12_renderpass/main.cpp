// Example 12 — gfx::RenderPass: offscreen render-to-texture with blit to screen.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
uniform mat4 u_view_proj;
uniform float u_time;
out float v_brightness;
void main() {
    float angle = u_time * 0.5;
    float ca = cos(angle), sa = sin(angle);
    vec3 p = vec3(a_pos.x * ca - a_pos.z * sa, a_pos.y, a_pos.x * sa + a_pos.z * ca);
    gl_Position = u_view_proj * vec4(p, 1.0);
    v_brightness = 0.4 + 0.6 * abs(a_normal.y);
}
)";

static const char* frag_src = R"(
#version 460 core
uniform vec4 u_color;
in float v_brightness;
out vec4 frag_color;
void main() {
    frag_color = vec4(u_color.rgb * v_brightness, u_color.a);
}
)";

int main() {
    gfx::Window window({"12 RenderPass", 800, 600});

    // --- Cube mesh ---
    std::vector<gfx::Vertex> verts;
    std::vector<unsigned int> idx;
    auto face = [&](float cx, float cy, float cz, float nx, float ny, float nz,
                    float ux, float uy, float uz, float vx, float vy, float vz) {
        auto base = static_cast<unsigned int>(verts.size());
        float h = 1.0f;
        verts.push_back({{cx - ux*h - vx*h, cy - uy*h - vy*h, cz - uz*h - vz*h}, {nx, ny, nz}, {0,0}, {}});
        verts.push_back({{cx + ux*h - vx*h, cy + uy*h - vy*h, cz + uz*h - vz*h}, {nx, ny, nz}, {1,0}, {}});
        verts.push_back({{cx + ux*h + vx*h, cy + uy*h + vy*h, cz + uz*h + vz*h}, {nx, ny, nz}, {1,1}, {}});
        verts.push_back({{cx - ux*h + vx*h, cy - uy*h + vy*h, cz - uz*h + vz*h}, {nx, ny, nz}, {0,1}, {}});
        idx.insert(idx.end(), {base, base+1, base+2, base, base+2, base+3});
    };
    face( 0, 0, 1,  0, 0, 1,  1,0,0,  0,1,0);
    face( 0, 0,-1,  0, 0,-1, -1,0,0,  0,1,0);
    face( 1, 0, 0,  1, 0, 0,  0,0,-1, 0,1,0);
    face(-1, 0, 0, -1, 0, 0,  0,0,1,  0,1,0);
    face( 0, 1, 0,  0, 1, 0,  1,0,0,  0,0,1);
    face( 0,-1, 0,  0,-1, 0,  1,0,0,  0,0,-1);

    gfx::Mesh cube;
    cube.set_vertices(verts);
    cube.set_indices(idx);
    cube.build();

    // --- Shader ---
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    auto prog = std::make_unique<gl::Program>();
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) return EXIT_FAILURE;

    // --- Offscreen render target ---
    const int RT_SIZE = 512;
    gl::Texture rt_tex(gl::TextureType::tex_2d);
    rt_tex.image_2d(0, GL_RGBA8, RT_SIZE, RT_SIZE, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    rt_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    rt_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    gl::Renderbuffer depth_rbo;
    depth_rbo.storage(GL_DEPTH_COMPONENT24, RT_SIZE, RT_SIZE);

    auto offscreen_fb = std::make_shared<gl::Framebuffer>();
    offscreen_fb->attach_texture(GL_COLOR_ATTACHMENT0, rt_tex);
    offscreen_fb->attach_renderbuffer(GL_DEPTH_ATTACHMENT, depth_rbo);
    offscreen_fb->check();

    // --- Material ---
    gfx::Material mat;
    mat.set_program(std::move(prog));

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({3, 2, 4}, {0, 0, 0});

    // --- Renderer ---
    gfx::Renderer renderer;
    gl::enable(GL_DEPTH_TEST);

    // --- Offscreen RenderPass ---
    gfx::RenderPass::Desc off_desc;
    off_desc.framebuffer = offscreen_fb;
    off_desc.clear_color = {0.05f, 0.0f, 0.0f, 1.0f};
    off_desc.clear_depth_attachment = true;
    gfx::RenderPass off_pass(off_desc);
    off_pass.set_viewport(0, 0, RT_SIZE, RT_SIZE);

    while (!window.should_close()) {
        float now = window.time();
        cam.set_aspect(float(window.width()) / window.height());
        mat.set_uniform("u_view_proj", cam.view_projection());
        mat.set_uniform("u_time", now);

        // Pass 1: render red cube to offscreen fb
        off_pass.begin();
        mat.set_uniform("u_color", glm::vec4(0.9f, 0.2f, 0.2f, 1.0f));
        renderer.draw(cube, mat);
        off_pass.end();

        // Blit offscreen fb to left half of screen
        int w = window.framebuffer_width();
        int h = window.framebuffer_height();
        offscreen_fb->blit_to(0, 0, 0, RT_SIZE, RT_SIZE, 0, 0, w / 2, h,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR);

        // Pass 2: render blue cube to default fb (right half)
        renderer.set_clear_color(0.0f, 0.0f, 0.05f, 1.0f);
        gl::enable(GL_SCISSOR_TEST);
        gl::scissor(w / 2, 0, w / 2, h);
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl::disable(GL_SCISSOR_TEST);

        renderer.viewport(w / 2, 0, w / 2, h);
        mat.set_uniform("u_color", glm::vec4(0.2f, 0.6f, 0.9f, 1.0f));
        renderer.draw(cube, mat);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
