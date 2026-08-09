// Example 31 — Mesh → Occlusion-Free Patch Atlas Pipeline
//
// Thin driver for gfx::CoverageAtlas (gllib/gfx/coverage_atlas.cpp): welds and
// clusters a loaded model into occlusion-free triangle patches, packs them
// into a texture atlas, builds a BVH, rasterises depth/thickness/UV coverage
// and emits three adaptive MIP4 sparse mip-chains, then renders the result
// using OpenGL multi-draw indirect with per-patch colours.
//
// Produces (in the output directory, default "."):
//   patch_table.bin        SoA patch metadata (v2: + per-patch double_sided)
//   patch_tris.bin         Reordered index buffer, patch-major
//   bvh_nodes.bin          Flattened BVH with 16-bit quantised AABBs
//   atlas_depth.bin        Adaptive mip-chain depth map   (RG8  min/max per node)
//   atlas_thickness.bin    Adaptive mip-chain thickness   (R8   max per node)
//   atlas_uv.bin           Adaptive mip-chain UV map      (RG8  average per node)
//   patch_summary.txt      Human-readable statistics
//
// Each atlas file is a self-describing sparse mip chain (header + levels).
// Level 0 has one node per patch at the same index the patch occupies in
// patch_table.bin / patch_tris.bin / the BVH (pack order); every level is a pair
// of plain GPU textures (a value texture + a u32 traversal-metadata texture),
// so coarse->fine traversal is a short chain of native texture fetches.
// Nodes subdivide 1→4 while their region is partially covered (so coverage
// silhouettes stay sharp to the finest level) or while their depth/thickness/UV
// range exceeds a small tolerance, so flat regions (walls, floors) stay at a few
// coarse nodes while detailed regions keep fine resolution. See docs/mip4.md.
//
// Usage: 31_mesh_decomposition [model.glb] [texel_density] [budget_Mtexels]
//        [min_patch_size] [epsilon] [axis_threshold] [min_tex] [max_tex]
//        [mip_tol_frac] [mip_leaf_tile]
//
// texel_density: texels per world unit (primary resolution control; 0 = auto,
//                sized so the model's longest axis gets `auto_target` texels).
// mip_leaf_tile: finest mip node size in texels (default 4; 1 = per-texel
//                detail on silhouettes and complex geometry).

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <functional>

// ==========================================================================
// §9  GLSL Shaders + OpenGL Multi-Draw Indirect Rendering
// ==========================================================================

static const char* vert_src = R"(
#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_vp;
out vec3 v_normal;
flat out int v_patch_id;
void main() {
    gl_Position = u_vp * vec4(a_pos, 1.0);
    v_normal = a_normal;
    v_patch_id = gl_DrawID;
}
)";

static const char* frag_src = R"(
#version 460 core
in vec3 v_normal;
flat in int v_patch_id;
layout(std430, binding = 0) readonly buffer ColorBuf {
    vec4 patch_colors[];
};
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(1.0, 2.0, 1.5));
    // Two-sided-ish lighting with a high ambient so the far side stays visible.
    float diff = abs(dot(n, light_dir));
    float light = 0.55 + 0.45 * diff;
    vec3 base_color = patch_colors[v_patch_id].rgb;
    frag_color = vec4(base_color * light, 1.0);
}
)";

struct RenderVertex { float position[3]; float normal[3]; };

int main(int argc, char** argv) {
    const char* model_path = "Stanford_Dragon.glb";
    gfx::CoverageAtlasConfig cfg;
    const char* out_dir = ".";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) cfg.texel_density = std::atof(argv[2]);
    if (argc > 3) cfg.budget_texels = std::atof(argv[3]) * 1e6f;
    if (argc > 4) cfg.min_patch_size = std::atoi(argv[4]);
    if (argc > 5) cfg.epsilon = std::atof(argv[5]);
    if (argc > 6) cfg.axis_threshold = std::atof(argv[6]);
    if (argc > 7) cfg.min_tex = std::atoi(argv[7]);
    if (argc > 8) cfg.max_tex = std::atoi(argv[8]);
    if (argc > 9) cfg.mip_tol_frac = std::atof(argv[9]);
    if (argc > 10) cfg.mip_leaf_tile = std::atoi(argv[10]);

    // --- Create window first (needed for GL context for model loading) ---
    gfx::Window window({"31 Mesh Decomposition", 1280, 720});

    // --- Load mesh via gfx::Model ---
    gfx::Model model;
    if (!model.load(model_path)) {
        fprintf(stderr, "Error: failed to load %s\n", model_path);
        return 1;
    }
    printf("Loaded: %zu meshes\n", model.mesh_count());

    // --- Run the coverage-atlas pipeline (CPU only) ---
    gfx::CoverageAtlas atlas(cfg);
    if (!atlas.build(model)) {
        fprintf(stderr, "Error: model has no geometry\n");
        return 1;
    }
    if (!atlas.write_files(out_dir)) {
        fprintf(stderr, "Error: failed to write outputs to %s\n", out_dir);
        return 1;
    }

    printf("Rendering with OpenGL (left-drag=orbit, scroll=zoom)...\n");

    const auto& positions = atlas.positions();
    const auto& normals   = atlas.normals();
    const auto& triangles = atlas.triangles();
    const auto& patches   = atlas.patches();

    // Build render vertices (position + normal)
    std::vector<RenderVertex> render_verts(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        render_verts[i] = {
            {positions[i].x, positions[i].y, positions[i].z},
            {normals[i].x, normals[i].y, normals[i].z}
        };
    }

    // Build patch-major index buffer
    std::vector<GLuint> render_indices;
    render_indices.reserve(triangles.size() * 3);
    for (auto& p : patches) {
        for (int ti : p.tris) {
            render_indices.push_back(triangles[size_t(ti)].v[0]);
            render_indices.push_back(triangles[size_t(ti)].v[1]);
            render_indices.push_back(triangles[size_t(ti)].v[2]);
        }
    }

    // Build indirect draw commands.
    // NOTE: DrawElementsIndirectCommand.firstIndex is an ELEMENT offset (the
    // driver multiplies by the index size internally), not a byte offset.
    std::vector<gl::DrawElementsIndirectCommand> draw_cmds;
    {
        GLuint index_offset = 0;
        for (auto& p : patches) {
            GLuint count = GLuint(p.tris.size() * 3);
            draw_cmds.push_back({count, 1, index_offset, 0, 0});
            index_offset += count;
        }
    }

    // Per-patch colours (random, well-separated)
    struct alignas(16) PatchColor { float r, g, b, a; };
    std::vector<PatchColor> patch_colors(patches.size());
    {
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> s_dist(0.45f, 1.0f);
        std::uniform_real_distribution<float> v_dist(0.7f, 1.0f);
        auto rand_hue = [&rng]() { return std::fmod(std::fmod(float(rng()) * 2.3283064e-10f, 1.0f) + 1.0f, 1.0f); };
        for (size_t i = 0; i < patches.size(); ++i) {
            float h = rand_hue();
            float s = s_dist(rng);
            float v = v_dist(rng);
            // HSV -> RGB
            float r, g, b;
            float c = v * s;
            float hp = h * 6.0f;
            float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
            switch (int(hp)) {
                case 0: r = c; g = x; b = 0; break;
                case 1: r = x; g = c; b = 0; break;
                case 2: r = 0; g = c; b = x; break;
                case 3: r = 0; g = x; b = c; break;
                case 4: r = x; g = 0; b = c; break;
                default: r = c; g = 0; b = x; break;
            }
            float m = v - c;
            patch_colors[i] = {r + m, g + m, b + m, 1.0f};
        }
    }

    // Create GL objects
    GLuint vao, vbo, ebo, indirect_buf, color_ssbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glGenBuffers(1, &indirect_buf);
    glGenBuffers(1, &color_ssbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, render_verts.size() * sizeof(RenderVertex),
                 render_verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, render_indices.size() * sizeof(GLuint),
                 render_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex),
                          (void*)offsetof(RenderVertex, normal));

    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, draw_cmds.size() * sizeof(gl::DrawElementsIndirectCommand),
                 draw_cmds.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, color_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, patch_colors.size() * sizeof(PatchColor),
                 patch_colors.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, color_ssbo);

    glBindVertexArray(0);

    // Compile shaders
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) {
        fprintf(stderr, "Vertex shader failed:\n%s\n", vs.info_log().c_str());
        return 1;
    }
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) {
        fprintf(stderr, "Fragment shader failed:\n%s\n", fs.info_log().c_str());
        return 1;
    }
    gl::Program prog;
    prog.attach(vs);
    prog.attach(fs);
    if (!prog.link()) {
        fprintf(stderr, "Program link failed:\n%s\n", prog.info_log().c_str());
        return 1;
    }

    // Camera
    glm::vec3 scene_lo(1e30f), scene_hi(-1e30f);
    for (auto& p : positions) {
        scene_lo = glm::min(scene_lo, p);
        scene_hi = glm::max(scene_hi, p);
    }
    glm::vec3 scene_center = (scene_lo + scene_hi) * 0.5f;
    float scene_radius = glm::length(scene_hi - scene_lo) * 0.5f;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / float(window.height()),
                    scene_radius * 0.001f, scene_radius * 10.0f);
    cam.look_at(
        glm::vec3(scene_center.x + scene_radius * 2.0f,
                   scene_center.y + scene_radius * 1.0f,
                   scene_center.z + scene_radius * 2.0f),
        scene_center
    );

    double prev_x, prev_y;
    window.cursor_position(prev_x, prev_y);

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);

    printf("Rendering %zu patches, %zu draw commands\n",
           patches.size(), draw_cmds.size());

    while (!window.should_close()) {
        window.poll_events();

        // Orbit (left drag)
        if (window.mouse_down(gfx::MouseButton::left)) {
            double cx, cy;
            window.cursor_position(cx, cy);
            float dx = float(cx - prev_x) * 0.005f;
            float dy = float(prev_y - cy) * 0.005f;
            cam.orbit(dx, dy);
            prev_x = cx;
            prev_y = cy;
        } else {
            window.cursor_position(prev_x, prev_y);
        }

        // Zoom (scroll)
        double scroll = window.scroll_delta();
        if (scroll != 0.0) cam.zoom(float(scroll) * scene_radius * 0.01f);

        cam.set_aspect(float(window.width()) / float(window.height()));

        // Render
        gl::viewport(0, 0, window.width(), window.height());
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        prog.use();
        glm::mat4 vp = cam.view_projection();
        glUniformMatrix4fv(prog.uniform_location("u_vp"), 1, GL_FALSE, &vp[0][0]);

        glBindVertexArray(vao);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, color_ssbo);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf);
        gl::multi_draw_elements_indirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                          nullptr, GLsizei(draw_cmds.size()), 0);

        window.swap_buffers();
    }

    return 0;
}
