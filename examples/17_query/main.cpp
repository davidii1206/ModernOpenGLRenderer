// Example 17 — gl::Query: GPU timer + occlusion queries.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)";

static const char* frag_src = R"(
#version 460 core
uniform vec4 u_color;
out vec4 frag_color;
void main() {
    frag_color = u_color;
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
    gfx::Window window({"17 Query", 800, 600});

    gfx::Mesh big_cube, small_cube;
    build_cube(big_cube);
    build_cube(small_cube);

    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    gl::Program prog;
    prog.attach(vs);
    prog.attach(fs);
    if (!prog.link()) return EXIT_FAILURE;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({0, 0, 6}, {0, 0, 0});

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    // Timer + occlusion queries (ping-pong two occlusion queries to avoid stalling)
    gl::Query timer(gl::QueryType::time_elapsed);
    gl::Query occ_a(gl::QueryType::any_samples_passed);
    gl::Query occ_b(gl::QueryType::any_samples_passed);
    gl::Query* occ_current = &occ_a;
    gl::Query* occ_prev    = &occ_b;

    bool small_visible = true;
    int frame = 0;

    while (!window.should_close()) {
        float now = window.time();
        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());

        glm::mat4 view_proj = cam.view_projection();
        glm::mat4 big_mvp = view_proj * glm::scale(glm::translate(glm::mat4(1), glm::vec3(0, 0, 0)), glm::vec3(1.5f));

        float sx = 2.2f * cos(now * 0.6f);
        float sz = 2.2f * sin(now * 0.6f);
        glm::mat4 small_mvp_base = glm::translate(glm::mat4(1), glm::vec3(sx, 0.0f, sz));

        // Read previous frame's occlusion result
        if (occ_prev->result_available()) {
            small_visible = occ_prev->result() > 0;
        }

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        timer.begin();
        prog.use();

        // Big cube (always drawn)
        GLint loc_mvp = prog.uniform_location("u_mvp");
        if (loc_mvp >= 0) prog.uniform_matrix4fv(loc_mvp, &big_mvp[0][0]);
        GLint loc_col = prog.uniform_location("u_color");
        if (loc_col >= 0) prog.uniform4f(loc_col, 0.3f, 0.5f, 0.9f, 1.0f);
        big_cube.draw();

        // Small cube (conditionally drawn based on last frame's occlusion)
        if (small_visible) {
            occ_current->begin();
            if (loc_mvp >= 0) prog.uniform_matrix4fv(loc_mvp, &small_mvp_base[0][0]);
            if (loc_col >= 0) prog.uniform4f(loc_col, 0.2f, 0.9f, 0.3f, 1.0f);
            small_cube.draw();
            occ_current->end();
        } else {
            // Draw a small indicator when occluded (subtle)
            glm::mat4 indicator = view_proj * glm::scale(
                glm::translate(glm::mat4(1), glm::vec3(sx, -1.8f, 0)), glm::vec3(0.1f));
            if (loc_mvp >= 0) prog.uniform_matrix4fv(loc_mvp, &indicator[0][0]);
            if (loc_col >= 0) prog.uniform4f(loc_col, 0.4f, 0.1f, 0.1f, 0.5f);
            small_cube.draw();
        }

        timer.end();

        // Swap occlusion queries
        std::swap(occ_current, occ_prev);

        if (frame % 60 == 0) {
            GLuint64 ns = timer.result();
            fprintf(stderr, "Frame %d — GPU: %.3f ms | small cube: %s\n",
                    frame, double(ns) / 1e6,
                    small_visible ? "VISIBLE" : "OCCLUDED");
        }

        window.swap_buffers();
        window.poll_events();
        ++frame;
    }

    return EXIT_SUCCESS;
}
