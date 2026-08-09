// Example 13 — gfx::Scene / gfx::Node: transform hierarchy with orbiting children.

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
uniform vec4 u_color;
out float v_brightness;
out vec4 v_color;
void main() {
    gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
    v_brightness = 0.4 + 0.6 * abs(a_normal.y);
    v_color = u_color;
}
)";

static const char* frag_src = R"(
#version 460 core
in float v_brightness;
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color.rgb * v_brightness, v_color.a);
}
)";

int main() {
    gfx::Window window({"13 Scene", 800, 600});

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

    // --- Material ---
    gfx::Material mat;
    mat.set_program(std::move(prog));

    // --- Scene graph ---
    gfx::Scene scene;
    auto* root = scene.root();

    auto* parent_node = root->create_child("parent");
    parent_node->set_mesh(&cube);
    parent_node->set_material(&mat);

    auto* child1 = parent_node->create_child("child1");
    child1->set_mesh(&cube);
    child1->set_material(&mat);
    child1->set_local_transform(glm::translate(glm::mat4(1), glm::vec3(2.5f, 1.5f, 0)));

    auto* child2 = parent_node->create_child("child2");
    child2->set_mesh(&cube);
    child2->set_material(&mat);
    child2->set_local_transform(glm::translate(glm::mat4(1), glm::vec3(-2.5f, -1.5f, 0)));

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({5, 3, 6}, {0, 0, 0});

    // --- Renderer ---
    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    // Colors per node (Scene::draw sets u_model, u_view_proj via material)
    auto draw_node = [&](gfx::Node* node, const glm::vec4& color) {
        mat.set_uniform("u_color", color);
        mat.set_uniform("u_view_proj", cam.view_projection());
        mat.set_uniform("u_model", node->world_transform());
        renderer.draw(*node->mesh(), mat);
    };

    float last_time = window.time();
    while (!window.should_close()) {
        float now = window.time();
        float dt = now - last_time;
        last_time = now;

        cam.set_aspect(float(window.width()) / window.height());

        // Animate: parent spins slowly, children orbit at different speeds
        float t = now * 0.6f;
        parent_node->set_local_transform(glm::rotate(glm::mat4(1), t, glm::vec3(0, 1, 0)));

        float t2 = now * 1.2f;
        child1->set_local_transform(
            glm::translate(glm::mat4(1), glm::vec3(2.5f * cos(t2), 1.5f * sin(t2), 1.0f * sin(t2 * 0.7f))));

        float t3 = now * -0.9f;
        child2->set_local_transform(
            glm::translate(glm::mat4(1), glm::vec3(2.5f * cos(t3), -1.5f * sin(t3), 1.0f * sin(t3 * 0.5f))));

        scene.update_transforms();

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        draw_node(parent_node, glm::vec4(0.8f, 0.3f, 0.8f, 1.0f));
        draw_node(child1, glm::vec4(0.2f, 0.8f, 0.3f, 1.0f));
        draw_node(child2, glm::vec4(0.9f, 0.6f, 0.2f, 1.0f));

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
