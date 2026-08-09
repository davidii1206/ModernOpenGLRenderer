// Example 21 — gfx::PostProcessing: HDR rendering, bloom, ACES tonemapping.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform vec3 u_color;
out float v_brightness;
out vec3 v_color;
void main() {
    gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
    v_brightness = 0.3 + 0.7 * abs(a_normal.y);
    v_color = u_color;
}
)";

static const char* frag_src = R"(
#version 460 core
in float v_brightness;
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color * v_brightness * 8.0, 1.0);
}
)";

static void build_cube(gfx::Mesh& mesh) {
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
    mesh.set_vertices(verts);
    mesh.set_indices(idx);
    mesh.build();
}

int main() {
    gfx::Window window({"21 PostProcessing", 800, 600});

    gfx::Mesh cube;
    build_cube(cube);

    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    auto prog = std::make_unique<gl::Program>();
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) return EXIT_FAILURE;

    gfx::Material mat;
    mat.set_program(std::move(prog));

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({4, 3, 5}, {0, 0, 0});

    gl::enable(GL_DEPTH_TEST);

    // Post-processing: HDR FBO + bloom + tonemapping
    gfx::PostProcessing pp(window.width(), window.height());
    pp.set_exposure(0.6f);
    pp.set_bloom_threshold(0.5f);
    pp.set_bloom_intensity(1.2f);

    while (!window.should_close()) {
        float now = window.time();
        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());
        pp.resize(window.width(), window.height());

        // Render scene into HDR framebuffer
        pp.begin(true);

        float t = now * 0.5f;
        for (int i = 0; i < 6; ++i) {
            float a = float(i) * 1.047f + t;
            float r = 2.5f;
            glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(cos(a) * r, sin(a * 0.7f) * 0.5f, sin(a) * r));
            model = glm::rotate(model, t + float(i), glm::vec3(0, 1, 0));
            model = glm::scale(model, glm::vec3(0.5f));

            glm::vec3 colors[] = {
                {8, 1, 1}, {1, 6, 1}, {1, 1, 8},
                {6, 6, 1}, {1, 6, 6}, {6, 1, 6},
            };

            mat.set_uniform("u_view_proj", cam.view_projection());
            mat.set_uniform("u_model", model);
            mat.set_uniform("u_color", colors[i % 6]);
            mat.bind();
            cube.draw();
        }

        pp.end();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
