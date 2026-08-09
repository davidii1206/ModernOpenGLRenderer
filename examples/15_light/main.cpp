// Example 15 — gfx::LightBuffer: directional + point lights with Blinn-Phong shading.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;

uniform mat4 u_view_proj;
uniform mat4 u_model;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world_pos;
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(u_model) * a_normal);
    v_uv = a_uv;
}
)";

static const char* frag_src = R"(
#version 460 core

#define MAX_DIR_LIGHTS 4
#define MAX_POINT_LIGHTS 16

struct DirectionalLight {
    vec4 direction;
    vec4 color;
};

struct PointLight {
    vec4 position;
    vec4 color;
    vec4 attenuation;
};

layout(std140, binding = 0) uniform LightBlock {
    DirectionalLight u_directional_lights[MAX_DIR_LIGHTS];
    PointLight u_point_lights[MAX_POINT_LIGHTS];
    int u_directional_light_count;
    int u_point_light_count;
    int _pad0;
    int _pad1;
};

in vec3 v_world_pos;
in vec3 v_normal;
out vec4 frag_color;

uniform vec3 u_view_pos;
uniform vec4 u_color = vec4(1.0);

vec3 calc_directional(DirectionalLight l, vec3 N, vec3 V) {
    vec3 L = normalize(-l.direction.xyz);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);
    return l.color.rgb * l.color.a * (diff + spec * 0.5);
}

vec3 calc_point(PointLight l, vec3 world_pos, vec3 N, vec3 V) {
    vec3 diff = l.position.xyz - world_pos;
    float dist = length(diff);
    if (dist > l.attenuation.w) return vec3(0.0);
    vec3 L = diff / dist;
    vec3 H = normalize(L + V);
    float atten = 1.0 / (l.attenuation.x + l.attenuation.y * dist + l.attenuation.z * dist * dist);
    float diff_factor = max(dot(N, L), 0.0);
    float spec_factor = pow(max(dot(N, H), 0.0), 64.0);
    return l.color.rgb * l.color.a * (diff_factor + spec_factor * 0.5) * atten;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_view_pos - v_world_pos);
    vec3 light_acc = vec3(0.05);

    for (int i = 0; i < u_directional_light_count; ++i)
        light_acc += calc_directional(u_directional_lights[i], N, V);

    for (int i = 0; i < u_point_light_count; ++i)
        light_acc += calc_point(u_point_lights[i], v_world_pos, N, V);

    frag_color = vec4(light_acc * u_color.rgb, u_color.a);
}
)";

int main() {
    gfx::Window window({"15 LightBuffer", 800, 600});

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

    // Light setup: one directional light + one animated point light
    gfx::LightBuffer lights;
    lights.set_directional(0, glm::vec3(-1, -1, -1), glm::vec3(1), 1.0f);
    lights.set_directional_count(1);
    lights.set_point(0, glm::vec3(3, 2, 0), glm::vec3(1, 0.2, 0.2), 2.0f, 8.0f);
    lights.set_point_count(1);
    lights.upload();

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({4, 3, 5}, {0, 0, 0});

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    while (!window.should_close()) {
        float now = window.time();
        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());

        float px = 3.0f * cos(now * 0.8f);
        float pz = 3.0f * sin(now * 0.8f);
        lights.set_point(0, glm::vec3(px, 1.5f + sin(now * 1.2f), pz),
                         glm::vec3(1, 0.2, 0.2), 2.0f, 8.0f);
        lights.upload();

        mat.set_uniform("u_view_proj", cam.view_projection());
        mat.set_uniform("u_model", glm::mat4(1.0f));
        mat.set_uniform("u_view_pos", cam.position());
        mat.set_uniform("u_color", glm::vec4(0.8f, 0.6f, 0.2f, 1.0f));

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        lights.bind(0);
        renderer.draw(cube, mat);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
