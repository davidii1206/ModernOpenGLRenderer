// Example 20 — gfx::Cubemap + gfx::Skybox: procedural skybox with reflective object.

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
uniform vec3 u_view_pos;

out vec3 v_normal;
out vec3 v_view_dir;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_normal = normalize(mat3(u_model) * a_normal);
    v_view_dir = normalize(u_view_pos - world.xyz);
}
)";

static const char* frag_src = R"(
#version 460 core
uniform samplerCube u_env_map;
uniform float u_fresnel_power;

in vec3 v_normal;
in vec3 v_view_dir;
out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(v_view_dir);
    vec3 R = reflect(-V, N);
    vec4 env = texture(u_env_map, R);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), u_fresnel_power + 0.001);
    frag_color = mix(env, vec4(1.0), fresnel * 0.4);
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
    gfx::Window window({"20 Environment", 800, 600});

    gfx::Cubemap env_map;
    env_map.generate_procedural(256);

    gfx::Skybox skybox(env_map);

    gfx::Mesh cube;
    build_cube(cube);

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
    cam.look_at({0, 1.5f, 4}, {0, 0, 0});

    gfx::Renderer renderer;
    renderer.set_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    auto uni_loc = [&](const char* name) { return prog.uniform_location(name); };

    while (!window.should_close()) {
        float now = window.time();
        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Skybox
        skybox.render(cam);

        // Reflective cube
        float angle = now * 0.5f;
        glm::mat4 model = glm::rotate(glm::mat4(1), angle, glm::vec3(0, 1, 0));

        prog.use();
        env_map.bind(0);

        GLint loc;
        loc = uni_loc("u_view_proj"); if (loc >= 0) prog.uniform_matrix4fv(loc, &cam.view_projection()[0][0]);
        loc = uni_loc("u_model");     if (loc >= 0) prog.uniform_matrix4fv(loc, &model[0][0]);
        loc = uni_loc("u_view_pos");  if (loc >= 0) prog.uniform3fv(loc, &cam.position()[0]);
        loc = uni_loc("u_env_map");   if (loc >= 0) prog.uniform1i(loc, 0);
        loc = uni_loc("u_fresnel_power"); if (loc >= 0) prog.uniform1f(loc, 3.0f);

        cube.draw();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
