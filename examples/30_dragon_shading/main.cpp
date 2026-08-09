// Example 30 â Stanford Dragon: triangle patch system with adjacency-based sorting,
// GPU patch detection via parallel prefix sum, indirect draw, and self-occlusion debug.


#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <imgui.h>
#include <gllib/log.hpp>


#include <glm/gtc/matrix_transform.hpp>
#include <stb_image_write.h>


#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_set>


// ---------------------------------------------------------------------------
// CPU preprocessing types
// ---------------------------------------------------------------------------


struct CPUTriangle {
   unsigned int v[3];
   int neighbor[3];
   int axis_id;
   int dominant_axis;
};


struct DrawCmd {
   unsigned int count;
   unsigned int instanceCount;
   unsigned int firstIndex;
   int baseVertex;
   unsigned int baseInstance;
};


struct AxisRange {
   int axis_id;
   int dominant_axis;
   unsigned int first_cmd;
   unsigned int cmd_count;
};


struct PatchMetaGPU {
   int32_t dominant_axis;
   int32_t atlas_x, atlas_y;
   int32_t atlas_w, atlas_h;
   float proj_origin_x, proj_origin_y;
   float height_min, height_max;
   int32_t _pad[3];
};
static_assert(sizeof(PatchMetaGPU) == 48, "PatchMetaGPU size");


// ---------------------------------------------------------------------------
// Edge adjacency helpers
// ---------------------------------------------------------------------------


struct EdgeEntry {
   int t0 = -1;
   int t1 = -1;
};


static uint64_t edge_key(unsigned int a, unsigned int b) {
   if (a > b) std::swap(a, b);
   return (uint64_t(a) << 32) | uint64_t(b);
}


static glm::vec3 axis_id_to_dir(int axis_id) {
   if (axis_id & 1)  return { 1,  0,  0};
   if (axis_id & 2)  return {-1,  0,  0};
   if (axis_id & 4)  return { 0,  1,  0};
   if (axis_id & 8)  return { 0, -1,  0};
   if (axis_id & 16) return { 0,  0,  1};
   if (axis_id & 32) return { 0,  0, -1};
   return {0, 1, 0};
}


static glm::vec3 perp_up(const glm::vec3& dir) {
   if (std::abs(dir.x) > 0.5f) return {0, 1, 0};
   if (std::abs(dir.y) > 0.5f) return {0, 0, 1};
   return {0, 1, 0};
}


// ---------------------------------------------------------------------------
// CPU preprocessing: adjacency, axis IDs, sorted order
// ---------------------------------------------------------------------------


static std::vector<CPUTriangle> build_preprocess(
   const gfx::Mesh& mesh,
   const glm::mat4& model_mat,
   glm::vec3& aabb_min_out,
   glm::vec3& aabb_max_out,
   std::vector<int>& split_out)
{
   const auto& vertices = mesh.vertices();
   const auto& indices = mesh.indices();
   size_t tri_count = indices.size() / 3;


   // Compute AABB
   aabb_min_out = glm::vec3( std::numeric_limits<float>::max());
   aabb_max_out = glm::vec3(-std::numeric_limits<float>::max());
   for (const auto& v : vertices) {
       aabb_min_out = glm::min(aabb_min_out, glm::vec3(v.position[0], v.position[1], v.position[2]));
       aabb_max_out = glm::max(aabb_max_out, glm::vec3(v.position[0], v.position[1], v.position[2]));
   }


   // --- Pass 1: build edge map with both triangles per edge ---
   std::unordered_map<uint64_t, EdgeEntry> edge_map;
   edge_map.reserve(tri_count * 3);
   for (size_t i = 0; i < tri_count; ++i) {
       for (int e = 0; e < 3; ++e) {
           unsigned int a = indices[i * 3 + e];
           unsigned int b = indices[i * 3 + (e + 1) % 3];
           uint64_t key = edge_key(a, b);
           auto& entry = edge_map[key];
           if (entry.t0 == -1)
               entry.t0 = int(i);
           else
               entry.t1 = int(i);
       }
   }


   // --- Pass 2: fill triangles with neighbors and axis IDs ---
   std::vector<CPUTriangle> tris(tri_count);
   for (size_t i = 0; i < tri_count; ++i) {
       tris[i].v[0] = indices[i * 3 + 0];
       tris[i].v[1] = indices[i * 3 + 1];
       tris[i].v[2] = indices[i * 3 + 2];


       for (int e = 0; e < 3; ++e) {
           unsigned int ea = tris[i].v[e];
           unsigned int eb = tris[i].v[(e + 1) % 3];
           uint64_t key = edge_key(ea, eb);
           auto it = edge_map.find(key);
           if (it != edge_map.end()) {
               auto& nb = it->second;
               tris[i].neighbor[e] = (nb.t0 == int(i)) ? nb.t1 : nb.t0;
           } else {
               tris[i].neighbor[e] = -1;
           }
       }


       // Axis bitmask from face normal â all significant axes
       glm::vec3 tp0(vertices[tris[i].v[0]].position[0],
                      vertices[tris[i].v[0]].position[1],
                      vertices[tris[i].v[0]].position[2]);
       glm::vec3 tp1(vertices[tris[i].v[1]].position[0],
                      vertices[tris[i].v[1]].position[1],
                      vertices[tris[i].v[1]].position[2]);
       glm::vec3 tp2(vertices[tris[i].v[2]].position[0],
                      vertices[tris[i].v[2]].position[1],
                      vertices[tris[i].v[2]].position[2]);
       glm::vec3 p0 = glm::vec3(model_mat * glm::vec4(tp0, 1.0f));
       glm::vec3 p1 = glm::vec3(model_mat * glm::vec4(tp1, 1.0f));
       glm::vec3 p2 = glm::vec3(model_mat * glm::vec4(tp2, 1.0f));
       glm::vec3 N = glm::normalize(glm::cross(p1 - p0, p2 - p0));
       int axis_id = 0;
       axis_id |= (N.x > 0.0f) ? 1 : 2;
       axis_id |= (N.y > 0.0f) ? 4 : 8;
       axis_id |= (N.z > 0.0f) ? 16 : 32;
       tris[i].axis_id = axis_id;


       float ax = std::abs(N.x), ay = std::abs(N.y), az = std::abs(N.z);
       int dominant_axis;
       if (ax >= ay && ax >= az)       dominant_axis = (N.x > 0) ? 1 : 2;
       else if (ay >= ax && ay >= az)  dominant_axis = (N.y > 0) ? 4 : 8;
       else                            dominant_axis = (N.z > 0) ? 16 : 32;
       tris[i].dominant_axis = dominant_axis;
   }


   // --- Sort: group by dominant_axis, BFS walk within each group ---
   int n = int(tri_count);
   std::vector<int> order(n);
   for (int i = 0; i < n; ++i) order[i] = i;
     std::sort(order.begin(), order.end(), [&](int a, int b) {
         return tris[a].dominant_axis != tris[b].dominant_axis
             ? tris[a].dominant_axis < tris[b].dominant_axis
             : tris[a].axis_id < tris[b].axis_id;
     });


   std::vector<int> old_to_new(n, -1);
   std::vector<bool> visited(n, false);
   std::vector<int> splits(n, 0);
   int sort_idx = 0;


     int g = 0;
     while (g < n) {
         int cur_da = tris[order[g]].dominant_axis;
         int cur_ai = tris[order[g]].axis_id;
         int g_end = g;
         while (g_end < n && tris[order[g_end]].dominant_axis == cur_da && tris[order[g_end]].axis_id == cur_ai)
             g_end++;


         for (int gi = g; gi < g_end; ++gi) {
             int start = order[gi];
             if (visited[start]) continue;
             // New connected component — mark end of previous component
             if (sort_idx > 0)
                 splits[sort_idx - 1] = 1;
             std::deque<int> frontier;
             frontier.push_back(start);
             visited[start] = true;
             while (!frontier.empty()) {
                 int cur = frontier.front();
                 frontier.pop_front();
                 old_to_new[cur] = sort_idx++;
                 for (int e = 0; e < 3; ++e) {
                     int nb = tris[cur].neighbor[e];
                     if (nb >= 0 && !visited[nb] && tris[nb].dominant_axis == cur_da && tris[nb].axis_id == cur_ai) {
                         visited[nb] = true;
                         frontier.push_back(nb);
                     }
                 }
             }
         }
         g = g_end;
     }
   // Mark last triangle as end of final patch
   if (sort_idx > 0)
       splits[sort_idx - 1] = 1;


   split_out = std::move(splits);


   // Invert and build sorted triangle list with remapped neighbors
   std::vector<int> new_to_old(n);
   for (int i = 0; i < n; ++i) new_to_old[old_to_new[i]] = i;


   std::vector<CPUTriangle> sorted(n);
   for (int new_id = 0; new_id < n; ++new_id) {
       int old_id = new_to_old[new_id];
       sorted[new_id] = tris[old_id];
       for (int e = 0; e < 3; ++e) {
           int old_nb = tris[old_id].neighbor[e];
           sorted[new_id].neighbor[e] = (old_nb >= 0) ? old_to_new[old_nb] : -1;
       }
   }


   gllib::logf(gllib::LogLevel::info, "preprocessed %d triangles, %zu edges (%d boundary)",
               n, edge_map.size(),
               int(std::count_if(edge_map.begin(), edge_map.end(),
                   [](const auto& kv) { return kv.second.t1 == -1; })));


   // Count adjacency groups
   std::unordered_map<int, int> axis_counts;
   for (auto& t : tris) axis_counts[t.dominant_axis]++;
   for (auto& [id, cnt] : axis_counts)
       gllib::logf(gllib::LogLevel::info, "  dominant_axis %d: %d triangles", id, cnt);


   return sorted;
}


// ---------------------------------------------------------------------------
// Compute shaders: split detection
// ---------------------------------------------------------------------------


static const char* split_detect_src = R"(
#version 460 core
layout(local_size_x = 256) in;


struct Triangle { int v[3]; int neighbor[3]; int axis_id; int dominant_axis; };


layout(std430, binding = 0) readonly buffer TriBuf { Triangle triangles[]; };
layout(std430, binding = 2) buffer SplitBuf { int splits[]; };


void main() {
   uint idx = gl_GlobalInvocationID.x;
   uint n = uint(triangles.length());
   if (idx >= n) return;


   splits[idx] = 0;


   if (idx == n - 1u) {
       splits[idx] = 1;
       return;
   }


   Triangle cur = triangles[idx];
   Triangle nxt = triangles[idx + 1u];


    if (cur.dominant_axis != nxt.dominant_axis || cur.axis_id != nxt.axis_id) {
        splits[idx] = 1;
        return;
    }


    bool connected = false;
    for (int e = 0; e < 3; e++) {
        int nb = nxt.neighbor[e];
        if (nb >= 0 && nb <= int(idx) && triangles[nb].dominant_axis == nxt.dominant_axis && triangles[nb].axis_id == nxt.axis_id) {
            connected = true;
            break;
        }
    }


   splits[idx] = connected ? 0 : 1;
}
)";






// ---------------------------------------------------------------------------
// Compute shaders: software rasterizer (per-patch)
// ---------------------------------------------------------------------------


static const char* rasterize_src = R"(
#version 460 core
layout(local_size_x = 256) in;


struct Triangle { int v[3]; int neighbor[3]; int axis_id; int dominant_axis; };
struct Vertex { float position[3]; float normal[3]; float texcoord[2]; float tangent[4]; };


struct PatchMeta {
   int  dominant_axis;
   int  atlas_x, atlas_y;
   int  atlas_w, atlas_h;
   float proj_origin_x, proj_origin_y;
   float height_min, height_max;
   int  _pad[3];
};


layout(std430, binding = 0) readonly buffer TriBuf  { Triangle triangles[]; };
layout(std430, binding = 1) readonly buffer VertBuf  { Vertex vertices[]; };
layout(std430, binding = 2) readonly buffer IdxBuf   { int patch_ids[]; };
layout(std430, binding = 3) readonly buffer PatchBuf { PatchMeta patches[]; };


layout(r32ui, binding = 0) uniform uimage2D u_height_atlas;


uniform uint   u_tri_count;
uniform int    u_patch_count;
uniform float  u_texel_density;
uniform float  u_axis_threshold;
uniform mat4   u_model;


vec3 dominant_dir(int axis) {
   if (axis == 1) return vec3( 1, 0, 0);
   if (axis == 2) return vec3(-1, 0, 0);
   if (axis == 4) return vec3( 0, 1, 0);
   if (axis == 8) return vec3( 0,-1, 0);
   if (axis == 16) return vec3( 0, 0, 1);
   return vec3( 0, 0,-1);
}


uint sortable_uint(float f) {
   uint u = floatBitsToUint(f);
   return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}


vec2 project(vec3 p, int axis) {
   if (axis == 1 || axis == 2)  return vec2(p.y, p.z);
   if (axis == 4 || axis == 8)  return vec2(p.x, p.z);
   return vec2(p.x, p.y);
}


float height_val(vec3 p, int axis) {
   if (axis == 1) return  p.x;
   if (axis == 2) return -p.x;
   if (axis == 4) return  p.y;
   if (axis == 8) return -p.y;
   if (axis == 16) return  p.z;
   return -p.z;
}


float edge(vec2 a, vec2 b, vec2 p) {
   return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}


void main() {
   uint t = gl_GlobalInvocationID.x;
   if (t >= u_tri_count) return;


   int pid = patch_ids[t];
   if (pid < 0 || pid >= u_patch_count) return;


   PatchMeta pm = patches[pid];
   if (pm.dominant_axis < 0) return;


   ivec3 vidx = ivec3(triangles[t].v[0], triangles[t].v[1], triangles[t].v[2]);
   vec3 lp0 = vec3(vertices[vidx.x].position[0], vertices[vidx.x].position[1], vertices[vidx.x].position[2]);
   vec3 lp1 = vec3(vertices[vidx.y].position[0], vertices[vidx.y].position[1], vertices[vidx.y].position[2]);
   vec3 lp2 = vec3(vertices[vidx.z].position[0], vertices[vidx.z].position[1], vertices[vidx.z].position[2]);
   vec3 p0 = vec3(u_model * vec4(lp0, 1.0));
   vec3 p1 = vec3(u_model * vec4(lp1, 1.0));
   vec3 p2 = vec3(u_model * vec4(lp2, 1.0));


   vec3 N = normalize(cross(p1 - p0, p2 - p0));
   if (abs(dot(N, dominant_dir(pm.dominant_axis))) < u_axis_threshold) return;


   float h0 = height_val(p0, pm.dominant_axis);
   float h1 = height_val(p1, pm.dominant_axis);
   float h2 = height_val(p2, pm.dominant_axis);


   vec2 proj0 = project(p0, pm.dominant_axis);
   vec2 proj1 = project(p1, pm.dominant_axis);
   vec2 proj2 = project(p2, pm.dominant_axis);


   ivec2 t0 = ivec2(int((proj0.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj0.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);
   ivec2 t1 = ivec2(int((proj1.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj1.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);
   ivec2 t2 = ivec2(int((proj2.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj2.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);


   ivec2 bb_min = min(t0, min(t1, t2));
   ivec2 bb_max = max(t0, max(t1, t2));
   ivec2 region_min = ivec2(pm.atlas_x, pm.atlas_y);
   ivec2 region_max = ivec2(pm.atlas_x + pm.atlas_w - 1, pm.atlas_y + pm.atlas_h - 1);
   bb_min = max(bb_min, region_min);
   bb_max = min(bb_max, region_max);
   if (bb_min.x > bb_max.x || bb_min.y > bb_max.y) return;


   vec2 f0 = vec2(t0);
   vec2 f1 = vec2(t1);
   vec2 f2 = vec2(t2);


   float area2 = edge(f0, f1, f2);
   if (abs(area2) < 1e-10) return;
   float inv_area2 = 1.0 / area2;


   for (int y = bb_min.y; y <= bb_max.y; ++y) {
       for (int x = bb_min.x; x <= bb_max.x; ++x) {
           vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
           float w0 = edge(f1, f2, p) * inv_area2;
           float w1 = edge(f2, f0, p) * inv_area2;
           float w2 = edge(f0, f1, p) * inv_area2;


           if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
               float h = w0 * h0 + w1 * h1 + w2 * h2;
               uint su = sortable_uint(h);
               imageAtomicMax(u_height_atlas, ivec2(x, y), su);
           }
       }
   }
}
)";


// ---------------------------------------------------------------------------
// Compute shader: occlusion detection (re-rasterize against finalized height atlas)
// ---------------------------------------------------------------------------


static const char* occlusion_detect_src = R"(
#version 460 core
layout(local_size_x = 256) in;


struct Triangle { int v[3]; int neighbor[3]; int axis_id; int dominant_axis; };
struct Vertex { float position[3]; float normal[3]; float texcoord[2]; float tangent[4]; };


struct PatchMeta {
   int  dominant_axis;
   int  atlas_x, atlas_y;
   int  atlas_w, atlas_h;
   float proj_origin_x, proj_origin_y;
   float height_min, height_max;
   int  _pad[3];
};


layout(std430, binding = 0) readonly buffer TriBuf  { Triangle triangles[]; };
layout(std430, binding = 1) readonly buffer VertBuf  { Vertex vertices[]; };
layout(std430, binding = 2) readonly buffer IdxBuf   { int patch_ids[]; };
layout(std430, binding = 3) readonly buffer PatchBuf { PatchMeta patches[]; };
layout(std430, binding = 4) writeonly buffer TriFlag { int tri_occluded[]; };


layout(r32ui, binding = 0) uniform readonly uimage2D u_height_atlas;
layout(r8,   binding = 1) uniform writeonly image2D u_occlusion_atlas;


uniform uint   u_tri_count;
uniform int    u_patch_count;
uniform float  u_texel_density;
uniform float  u_axis_threshold;
uniform mat4   u_model;


vec3 dominant_dir(int axis) {
   if (axis == 1) return vec3( 1, 0, 0);
   if (axis == 2) return vec3(-1, 0, 0);
   if (axis == 4) return vec3( 0, 1, 0);
   if (axis == 8) return vec3( 0,-1, 0);
   if (axis == 16) return vec3( 0, 0, 1);
   return vec3( 0, 0,-1);
}


uint sortable_uint(float f) {
   uint u = floatBitsToUint(f);
   return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}


float from_sortable(uint u) {
   if ((u & 0x80000000u) != 0)
       return uintBitsToFloat(u ^ 0x80000000u);
   else
       return uintBitsToFloat(~u);
}


vec2 project(vec3 p, int axis) {
   if (axis == 1 || axis == 2)  return vec2(p.y, p.z);
   if (axis == 4 || axis == 8)  return vec2(p.x, p.z);
   return vec2(p.x, p.y);
}


float height_val(vec3 p, int axis) {
   if (axis == 1) return  p.x;
   if (axis == 2) return -p.x;
   if (axis == 4) return  p.y;
   if (axis == 8) return -p.y;
   if (axis == 16) return  p.z;
   return -p.z;
}


float edge(vec2 a, vec2 b, vec2 p) {
   return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}


void main() {
   uint t = gl_GlobalInvocationID.x;
   if (t >= u_tri_count) return;


   tri_occluded[t] = 0;


   int pid = patch_ids[t];
   if (pid < 0 || pid >= u_patch_count) return;


   PatchMeta pm = patches[pid];
   if (pm.dominant_axis < 0) return;


   ivec3 vidx = ivec3(triangles[t].v[0], triangles[t].v[1], triangles[t].v[2]);
   vec3 lp0 = vec3(vertices[vidx.x].position[0], vertices[vidx.x].position[1], vertices[vidx.x].position[2]);
   vec3 lp1 = vec3(vertices[vidx.y].position[0], vertices[vidx.y].position[1], vertices[vidx.y].position[2]);
   vec3 lp2 = vec3(vertices[vidx.z].position[0], vertices[vidx.z].position[1], vertices[vidx.z].position[2]);
   vec3 p0 = vec3(u_model * vec4(lp0, 1.0));
   vec3 p1 = vec3(u_model * vec4(lp1, 1.0));
   vec3 p2 = vec3(u_model * vec4(lp2, 1.0));


   vec3 N = normalize(cross(p1 - p0, p2 - p0));
   if (abs(dot(N, dominant_dir(pm.dominant_axis))) < u_axis_threshold) return;


   float h0 = height_val(p0, pm.dominant_axis);
   float h1 = height_val(p1, pm.dominant_axis);
   float h2 = height_val(p2, pm.dominant_axis);


   vec2 proj0 = project(p0, pm.dominant_axis);
   vec2 proj1 = project(p1, pm.dominant_axis);
   vec2 proj2 = project(p2, pm.dominant_axis);


   ivec2 t0 = ivec2(int((proj0.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj0.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);
   ivec2 t1 = ivec2(int((proj1.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj1.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);
   ivec2 t2 = ivec2(int((proj2.x - pm.proj_origin_x) * u_texel_density + 0.5) + pm.atlas_x,
                     int((proj2.y - pm.proj_origin_y) * u_texel_density + 0.5) + pm.atlas_y);


   ivec2 bb_min = min(t0, min(t1, t2));
   ivec2 bb_max = max(t0, max(t1, t2));
   ivec2 region_min = ivec2(pm.atlas_x, pm.atlas_y);
   ivec2 region_max = ivec2(pm.atlas_x + pm.atlas_w - 1, pm.atlas_y + pm.atlas_h - 1);
   bb_min = max(bb_min, region_min);
   bb_max = min(bb_max, region_max);
   if (bb_min.x > bb_max.x || bb_min.y > bb_max.y) return;


   vec2 f0 = vec2(t0);
   vec2 f1 = vec2(t1);
   vec2 f2 = vec2(t2);


   float area2 = edge(f0, f1, f2);
   if (abs(area2) < 1e-10) return;
   float inv_area2 = 1.0 / area2;


   for (int y = bb_min.y; y <= bb_max.y; ++y) {
       for (int x = bb_min.x; x <= bb_max.x; ++x) {
           vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
           float w0 = edge(f1, f2, p) * inv_area2;
           float w1 = edge(f2, f0, p) * inv_area2;
           float w2 = edge(f0, f1, p) * inv_area2;


           if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
               float h = w0 * h0 + w1 * h1 + w2 * h2;
               uint su = sortable_uint(h);
               uint stored = imageLoad(u_height_atlas, ivec2(x, y)).r;
               if (stored != su) {
                   float stored_h = from_sortable(stored);
                   float my_h = from_sortable(su);
                   if (abs(stored_h - my_h) > 1e-3) {
                       imageStore(u_occlusion_atlas, ivec2(x, y), vec4(1.0, 0.0, 0.0, 0.0));
                       tri_occluded[t] = 1;
                   }
               }
           }
       }
   }
}
)";


// ---------------------------------------------------------------------------
// Compute shader: thickness post-pass from atomic height atlas
// ---------------------------------------------------------------------------


static const char* thickness_src = R"(
#version 460 core
layout(local_size_x = 8, local_size_y = 8) in;


struct PatchMeta {
   int  dominant_axis;
   int  atlas_x, atlas_y;
   int  atlas_w, atlas_h;
   float proj_origin_x, proj_origin_y;
   float height_min, height_max;
   int  _pad[3];
};


layout(r32ui, binding = 0) uniform readonly  uimage2D u_height_atlas;
layout(rg16f, binding = 1) uniform writeonly image2D u_output_atlas;
layout(r8,    binding = 2) uniform readonly  image2D u_occlusion_atlas;


layout(std430, binding = 3) readonly buffer PatchBuf { PatchMeta patches[]; };


uniform int u_atlas_w;
uniform int u_atlas_h;
uniform int u_patch_count;
uniform float u_global_h_min;
uniform float u_global_h_range;


float from_sortable(uint u) {
   if ((u & 0x80000000u) != 0)
       return uintBitsToFloat(u ^ 0x80000000u);
   else
       return uintBitsToFloat(~u);
}


void main() {
   ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
   if (pixel.x >= u_atlas_w || pixel.y >= u_atlas_h) return;


   uint raw = imageLoad(u_height_atlas, pixel).r;
   if (raw == 0u) {
       imageStore(u_output_atlas, pixel, vec4(0.0, 0.0, 0.0, 0.0));
       return;
   }


   float center_h = from_sortable(raw);


   float occ = imageLoad(u_occlusion_atlas, pixel).r;


   uint raw_l = imageLoad(u_height_atlas, pixel + ivec2(-1, 0)).r;
   uint raw_r = imageLoad(u_height_atlas, pixel + ivec2( 1, 0)).r;
   uint raw_u = imageLoad(u_height_atlas, pixel + ivec2( 0,-1)).r;
   uint raw_d = imageLoad(u_height_atlas, pixel + ivec2( 0, 1)).r;
   float h_l = (raw_l == 0u) ? center_h : from_sortable(raw_l);
   float h_r = (raw_r == 0u) ? center_h : from_sortable(raw_r);
   float h_u = (raw_u == 0u) ? center_h : from_sortable(raw_u);
   float h_d = (raw_d == 0u) ? center_h : from_sortable(raw_d);


   float dh_dx = (h_r - h_l) * 0.5;
   float dh_dy = (h_d - h_u) * 0.5;
   float thickness = sqrt(dh_dx * dh_dx + dh_dy * dh_dy);


   float h_norm = (u_global_h_range > 1e-10) ? clamp((center_h - u_global_h_min) / u_global_h_range, 0.0, 1.0) : 0.0;
   float t_scale = max(u_global_h_range * 0.02, 1e-6);
   float t_norm = clamp(thickness / t_scale, 0.0, 1.0);


   imageStore(u_output_atlas, pixel, vec4(h_norm, t_norm, occ, 0.0));
}
)";


// ---------------------------------------------------------------------------
// Compute shader: reduce occlusion atlas to per-patch flags
// ---------------------------------------------------------------------------


static const char* occlusion_reduce_src = R"(
#version 460 core
layout(local_size_x = 256) in;


layout(r8, binding = 0) uniform readonly image2D u_occlusion_atlas;


struct PatchMeta {
   int  dominant_axis;
   int  atlas_x, atlas_y;
   int  atlas_w, atlas_h;
   float proj_origin_x, proj_origin_y;
   float height_min, height_max;
   int  _pad[3];
};


layout(std430, binding = 0) writeonly buffer FlagBuf { int has_occlusion[]; };
layout(std430, binding = 1) readonly buffer PatchBuf { PatchMeta patches[]; };


uniform int u_patch_count;


void main() {
   uint pid = gl_GlobalInvocationID.x;
   if (pid >= uint(u_patch_count)) return;


   PatchMeta pm = patches[pid];
   has_occlusion[pid] = 0;


   for (int y = pm.atlas_y; y < pm.atlas_y + pm.atlas_h; ++y) {
       for (int x = pm.atlas_x; x < pm.atlas_x + pm.atlas_w; ++x) {
           float occ = imageLoad(u_occlusion_atlas, ivec2(x, y)).r;
           if (occ > 0.0) {
               has_occlusion[pid] = 1;
               return;
           }
       }
   }
}
)";


// ---------------------------------------------------------------------------
// Render shaders
// ---------------------------------------------------------------------------


static const char* render_vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in int a_tri_id;


uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform int u_draw_offset;


out vec3 v_world_pos;
flat out int v_patch_id;
flat out int v_global_tri_id;


void main() {
   vec4 wp = u_model * vec4(a_pos, 1.0);
   gl_Position = u_view_proj * wp;
   v_world_pos = wp.xyz;
   v_patch_id = gl_DrawID + u_draw_offset;
   v_global_tri_id = a_tri_id;
}
)";


static const char* render_frag_src = R"(
#version 460 core


in vec3 v_world_pos;
flat in int v_patch_id;
flat in int v_global_tri_id;


uniform vec3 u_view_pos;
uniform vec3 u_light_dir;
uniform vec4 u_color;
uniform bool u_flat_shading;
uniform bool u_dominant_axis;
uniform bool u_debug_view;
uniform bool u_patch_colors;
uniform bool u_show_occlusion;
uniform int u_draw_offset;


layout(std430, binding = 4) readonly buffer TriFlag { int tri_occluded[]; };


out vec4 frag_color;


vec3 hash_color(uint id) {
   id = id ^ (id >> 16u);
   id *= 0x45d9f3bu;
   id = id ^ (id >> 16u);
   id *= 0x45d9f3bu;
   id = id ^ (id >> 16u);
   float r = float( id        & 0xFFu) / 255.0;
   float g = float((id >> 8)  & 0xFFu) / 255.0;
   float b = float((id >> 16) & 0xFFu) / 255.0;
   return vec3(r, g, b);
}


void main() {
   vec3 N = normalize(cross(dFdx(v_world_pos), dFdy(v_world_pos)));


   if (u_debug_view) {
       vec3 pc = hash_color(uint(v_patch_id) + 1u);
       frag_color = vec4(pc, 1.0);
       return;
   }


   vec3 base_color = u_color.rgb;


   if (u_patch_colors) {
       base_color = hash_color(uint(v_patch_id) + 1u);
   } else if (u_dominant_axis) {
       float ax = abs(N.x), ay = abs(N.y), az = abs(N.z);
       int dom;
       if (ax >= ay && ax >= az)       dom = (N.x > 0.0) ? 1 : 2;
       else if (ay >= ax && ay >= az)  dom = (N.y > 0.0) ? 4 : 8;
       else                            dom = (N.z > 0.0) ? 16 : 32;
       base_color = hash_color(uint(dom));
   }


   if (u_flat_shading) {
       vec3 L = normalize(-u_light_dir);
       float diff = max(dot(N, L), 0.0);
       float ambient = 0.35;
       frag_color = vec4(base_color * (ambient + diff * 0.8), u_color.a);
   } else {
       frag_color = vec4(base_color, u_color.a);
   }


   if (u_show_occlusion) {
       if (tri_occluded[v_global_tri_id] != 0)
           frag_color = mix(frag_color, vec4(1.0, 0.0, 0.0, 1.0), 0.6);
   }
}
)";


// (old hardware occlusion shaders removed — replaced by compute rasterizer above)


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


static gl::Program* make_compute(const char* src) {
   auto* p = new gl::Program();
   gl::Shader s(gl::ShaderType::compute, src);
   if (!s.compiled()) { delete p; return nullptr; }
   p->attach(s);
   if (!p->link()) { delete p; return nullptr; }
   return p;
}


static gl::Program* make_render_program(const char* vs, const char* fs) {
   auto* p = new gl::Program();
   gl::Shader v(gl::ShaderType::vertex, vs);
   if (!v.compiled()) { delete p; return nullptr; }
   gl::Shader f(gl::ShaderType::fragment, fs);
   if (!f.compiled()) { delete p; return nullptr; }
   p->attach(v);
   p->attach(f);
   if (!p->link()) { delete p; return nullptr; }
   return p;
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------


int main(int argc, char** argv) {
   gllib::log_to_stderr(gllib::LogLevel::info);


   gfx::Window window({"30 Dragon Patches", 1280, 720});
   window.vsync(false);


   // --- Load model ---
   const char* model_path = "Stanford_Dragon.glb";
   if (argc > 1) model_path = argv[1];


   gfx::Model model;
   if (!model.load(model_path)) {
       gllib::log(gllib::LogLevel::error, "failed to load model");
       return EXIT_FAILURE;
   }


   const auto& mesh = model.mesh(0);
   gllib::logf(gllib::LogLevel::info, "model: %zu verts, %zu indices (%zu tris)",
               mesh.vertices().size(), mesh.indices().size(), mesh.indices().size() / 3);


   glm::mat4 model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));


   // --- CPU preprocessing (runs once, never again) ---
   glm::vec3 aabb_min, aabb_max;
   std::vector<int> cpu_split_flags;
   auto cpu_tris = build_preprocess(mesh, model_mat, aabb_min, aabb_max, cpu_split_flags);
   size_t tri_count = cpu_tris.size();


   glm::vec3 aabb_center = (aabb_min + aabb_max) * 0.5f;
   float aabb_radius = glm::length(aabb_max - aabb_min) * 0.5f;


   // Sorted index buffer for rendering
   std::vector<unsigned int> sorted_indices(tri_count * 3);
   for (size_t i = 0; i < tri_count; ++i) {
       sorted_indices[i * 3 + 0] = cpu_tris[i].v[0];
       sorted_indices[i * 3 + 1] = cpu_tris[i].v[1];
       sorted_indices[i * 3 + 2] = cpu_tris[i].v[2];
   }


   // --- GPU buffers ---
   // binding 0: triangle data
   gl::Buffer tri_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
   tri_buf.data(cpu_tris.data(), tri_count * sizeof(CPUTriangle));


   // binding 1: patch count
   unsigned int zero32 = 0;
   gl::Buffer count_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   count_buf.data(&zero32, sizeof(unsigned int));


   // binding 2: split flags
   std::vector<int> split_init(tri_count, 0);
   gl::Buffer split_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   split_buf.data(split_init.data(), tri_count * sizeof(int));


   // binding 3: indirect draw commands
   size_t max_cmds = tri_count;
   std::vector<DrawCmd> cmds_init(max_cmds);
   gl::Buffer indirect_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   indirect_buf.data(cmds_init.data(), max_cmds * sizeof(DrawCmd));


   // CPU readback buffer for split flags
   std::vector<int> cpu_splits(tri_count, 0);


   // Sorted index buffer
   gl::Buffer sorted_ebo(gl::BufferType::index, gl::BufferUsage::static_draw);
   sorted_ebo.data(sorted_indices.data(), sorted_indices.size() * sizeof(unsigned int));


   // --- VAO for indirect draw (model VBO + sorted EBO) ---
   gl::VertexArray patch_vao;
   patch_vao.bind();
   glVertexArrayVertexBuffer(patch_vao.handle(), 0, mesh.vbo_handle(), 0, sizeof(gfx::Vertex));
   glVertexArrayAttribFormat(patch_vao.handle(), 0, 3, GL_FLOAT, GL_FALSE, offsetof(gfx::Vertex, position));
   glVertexArrayAttribBinding(patch_vao.handle(), 0, 0);
   glEnableVertexArrayAttrib(patch_vao.handle(), 0);
   glVertexArrayAttribFormat(patch_vao.handle(), 1, 3, GL_FLOAT, GL_FALSE, offsetof(gfx::Vertex, normal));
   glVertexArrayAttribBinding(patch_vao.handle(), 1, 0);
   glEnableVertexArrayAttrib(patch_vao.handle(), 1);
   glEnableVertexArrayAttrib(patch_vao.handle(), 2);
   glVertexArrayAttribBinding(patch_vao.handle(), 2, 2);
   glVertexArrayElementBuffer(patch_vao.handle(), sorted_ebo.handle());


   // --- Compute shader: split detection only ---
   std::unique_ptr<gl::Program> p_split(make_compute(split_detect_src));
   if (!p_split) {
       gllib::log(gllib::LogLevel::error, "compute shader compilation failed");
       return EXIT_FAILURE;
   }


   // --- Render shader ---
   std::unique_ptr<gl::Program> render_prog(make_render_program(render_vert_src, render_frag_src));
   if (!render_prog) { gllib::log(gllib::LogLevel::error, "render shader failed"); return EXIT_FAILURE; }


   // --- Software rasterizer compute shader ---
   std::unique_ptr<gl::Program> raster_prog(make_compute(rasterize_src));
   if (!raster_prog) { gllib::log(gllib::LogLevel::error, "rasterize compute failed"); return EXIT_FAILURE; }


   std::unique_ptr<gl::Program> occ_detect_prog(make_compute(occlusion_detect_src));
   if (!occ_detect_prog) { gllib::log(gllib::LogLevel::error, "occlusion detect compute failed"); return EXIT_FAILURE; }


   std::unique_ptr<gl::Program> thickness_prog(make_compute(thickness_src));
   if (!thickness_prog) { gllib::log(gllib::LogLevel::error, "thickness compute failed"); return EXIT_FAILURE; }


   std::unique_ptr<gl::Program> occ_reduce_prog(make_compute(occlusion_reduce_src));
   if (!occ_reduce_prog) { gllib::log(gllib::LogLevel::error, "occlusion reduce compute failed"); return EXIT_FAILURE; }


   // --- Atlas textures ---
   GLuint height_atlas_tex = 0, occlusion_atlas_tex = 0, output_atlas_tex = 0;
   int atlas_w = 0, atlas_h = 0;


   // binding 2: per-triangle patch_id SSBO (filled after patch detection)
   gl::Buffer patch_id_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);


   // binding 3: per-patch metadata SSBO (filled after atlas packing)
   gl::Buffer patch_meta_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);


   // Per-patch occlusion flags + filtered draw buffers
   gl::Buffer occ_flag_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   gl::Buffer filtered_indirect_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   gl::Buffer filtered_count_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   std::vector<int> cpu_occ_flags;
   unsigned int filtered_patch_count = 0;
   unsigned int total_draw_count = 0;


   // Per-triangle occlusion flags for 3D visualization
   gl::Buffer tri_occluded_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
   gl::Buffer patch_first_tri_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
   gl::Buffer tri_id_buf(gl::BufferType::vertex, gl::BufferUsage::static_draw);
   std::vector<int> cpu_tri_occluded;


   // --- Camera ---
   gfx::Camera cam;
   cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 10000.0f);
   cam.look_at({3, 2, 4}, {0, 0, 0});


   glm::vec3 light_dir(5.0f, -9.0f, -1.5f);


   // --- ImGui ---
   gfx::ImGuiOverlay gui;
   if (!gui.init(window)) { gllib::log(gllib::LogLevel::error, "ImGui init failed"); return EXIT_FAILURE; }


   // --- State ---
   gfx::Renderer renderer;
   renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
   gl::enable(GL_DEPTH_TEST);


   float color_val[3] = {1.0f, 1.0f, 1.0f};
   bool flat_shading = true;
   bool dominant_axis = false;
   bool debug_view = false;
   bool patch_colors = false;
   bool use_indirect = true;
   bool show_occluded_only = false;
   bool show_occlusion_3d = false;
    float axis_threshold = 0.707106781187f;
    float merge_epsilon = 0.95f;
    bool write_textures = false;


   bool patches_ready = false;
   unsigned int patch_count = 0;
   float gpu_time_us = 0.0f;
   std::vector<AxisRange> axis_ranges;


   gl::Query gpu_timer(gl::QueryType::time_elapsed);


   double prev_mx = 0, prev_my = 0;
   bool mouse_down = false;
   bool middle_mouse_down = false;


   // Uniform locations
   GLint loc_vp  = render_prog->uniform_location("u_view_proj");
   GLint loc_m   = render_prog->uniform_location("u_model");
   GLint loc_vp2 = render_prog->uniform_location("u_view_pos");
   GLint loc_ld  = render_prog->uniform_location("u_light_dir");
   GLint loc_c   = render_prog->uniform_location("u_color");
   GLint loc_fs  = render_prog->uniform_location("u_flat_shading");
   GLint loc_da  = render_prog->uniform_location("u_dominant_axis");
   GLint loc_dv  = render_prog->uniform_location("u_debug_view");
   GLint loc_pc  = render_prog->uniform_location("u_patch_colors");
   GLint loc_do  = render_prog->uniform_location("u_draw_offset");
   GLint loc_so  = render_prog->uniform_location("u_show_occlusion");


   // Rasterize compute uniforms
   GLint loc_r_tri_cnt = raster_prog->uniform_location("u_tri_count");
   GLint loc_r_pat_cnt = raster_prog->uniform_location("u_patch_count");
   GLint loc_r_density = raster_prog->uniform_location("u_texel_density");
   GLint loc_r_thresh  = raster_prog->uniform_location("u_axis_threshold");
   GLint loc_r_model   = raster_prog->uniform_location("u_model");


   // Occlusion detect compute uniforms
   GLint loc_o_tri_cnt = occ_detect_prog->uniform_location("u_tri_count");
   GLint loc_o_pat_cnt = occ_detect_prog->uniform_location("u_patch_count");
   GLint loc_o_density = occ_detect_prog->uniform_location("u_texel_density");
   GLint loc_o_thresh  = occ_detect_prog->uniform_location("u_axis_threshold");
   GLint loc_o_model   = occ_detect_prog->uniform_location("u_model");


   // Thickness compute uniforms
   GLint loc_t_aw  = thickness_prog->uniform_location("u_atlas_w");
   GLint loc_t_ah  = thickness_prog->uniform_location("u_atlas_h");
   GLint loc_t_pc  = thickness_prog->uniform_location("u_patch_count");
   GLint loc_t_hmn = thickness_prog->uniform_location("u_global_h_min");
   GLint loc_t_hrg = thickness_prog->uniform_location("u_global_h_range");


   // Occlusion reduce compute uniforms
   GLint loc_red_pc = occ_reduce_prog->uniform_location("u_patch_count");


   while (!window.should_close()) {
       window.poll_events();
       cam.set_aspect(float(window.width()) / window.height());


       // Orbital controls
       double mx, my;
       window.cursor_position(mx, my);
       if (window.mouse_down(gfx::MouseButton::left)) {
           if (mouse_down)
               cam.orbit(float((mx - prev_mx) * 0.005), float((my - prev_my) * 0.005));
           mouse_down = true;
       } else {
           mouse_down = false;
       }
       if (window.mouse_down(gfx::MouseButton::middle)) {
           if (middle_mouse_down)
               cam.pan(float((mx - prev_mx) * -0.003), float((my - prev_my) * 0.003));
           middle_mouse_down = true;
       } else {
           middle_mouse_down = false;
       }
       prev_mx = mx;
       prev_my = my;


       double scroll = window.scroll_delta();
       if (scroll != 0.0)
           cam.zoom(float(-scroll * 0.2));


       // --- ImGui ---
       gui.begin_frame();
       {
           ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
           ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_Once);
           ImGui::Begin("Dragon Patches", nullptr, ImGuiWindowFlags_NoSavedSettings);
           ImGui::Text("FPS: %.0f (%.2f ms)", ImGui::GetIO().Framerate,
                       1000.0f / ImGui::GetIO().Framerate);
           ImGui::Separator();


           ImGui::ColorEdit3("Color", color_val);
           ImGui::Checkbox("Flat Shading", &flat_shading);
           ImGui::Checkbox("Dominant Axis", &dominant_axis);
           ImGui::Checkbox("Patch Colors", &patch_colors);
           ImGui::Separator();


           if (ImGui::Button("Run Patch Detection")) {
               gpu_timer.begin();


               p_split->use();
               tri_buf.bind_base(0);
               split_buf.bind_base(2);
               gl::dispatch_compute(GLuint((tri_count + 255) / 256), 1, 1);
               gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);


               gpu_timer.end();
               gpu_time_us = float(gpu_timer.result()) / 1000.0f;


               glMemoryBarrier(GL_ALL_BARRIER_BITS);
               {
                   gl::Sync fence;
                   fence.client_wait(1000000000);
               }
               glGetNamedBufferSubData(split_buf.handle(), 0, tri_count * sizeof(int), cpu_splits.data());


               patch_count = 0;
               for (size_t i = 0; i < tri_count; ++i)
                   if (cpu_splits[i] == 1) patch_count++;


               std::vector<DrawCmd> cmds(patch_count);
               std::vector<int> tri_patch_ids(tri_count, -1);
               {
                   unsigned int cmd_idx = 0;
                   size_t patch_start = 0;
                   int cur_pid = 0;
                   for (size_t i = 0; i < tri_count; ++i) {
                       tri_patch_ids[i] = cur_pid;
                       if (cpu_splits[i] == 1) {
                           unsigned int count = unsigned((i + 1) - patch_start) * 3u;
                           if (cmd_idx < patch_count) {
                               cmds[cmd_idx].count = count;
                               cmds[cmd_idx].instanceCount = 1;
                               cmds[cmd_idx].firstIndex = unsigned(patch_start) * 3u;
                               cmds[cmd_idx].baseVertex = 0;
                               cmds[cmd_idx].baseInstance = 0;
                           }
                           cmd_idx++;
                           patch_start = i + 1;
                           cur_pid++;
                       }
                   }
                }


                 // --- Greedy epsilon merge: merge adjacent patches with similar normals ---
                unsigned int pre_merge_count = patch_count;
                {
                    std::vector<glm::vec3> patch_normal(patch_count, glm::vec3(0.0f));
                    std::vector<int> patch_size(patch_count, 0);

                    for (unsigned int i = 0; i < tri_count; ++i) {
                        int pid = tri_patch_ids[i];
                        if (pid < 0 || pid >= int(patch_count)) continue;
                        const auto& tri = cpu_tris[i];
                        glm::vec3 wp0 = glm::vec3(model_mat * glm::vec4(
                            mesh.vertices()[tri.v[0]].position[0],
                            mesh.vertices()[tri.v[0]].position[1],
                            mesh.vertices()[tri.v[0]].position[2], 1.0f));
                        glm::vec3 wp1 = glm::vec3(model_mat * glm::vec4(
                            mesh.vertices()[tri.v[1]].position[0],
                            mesh.vertices()[tri.v[1]].position[1],
                            mesh.vertices()[tri.v[1]].position[2], 1.0f));
                        glm::vec3 wp2 = glm::vec3(model_mat * glm::vec4(
                            mesh.vertices()[tri.v[2]].position[0],
                            mesh.vertices()[tri.v[2]].position[1],
                            mesh.vertices()[tri.v[2]].position[2], 1.0f));
                        glm::vec3 n = glm::cross(wp1 - wp0, wp2 - wp0);
                        float len = glm::length(n);
                        if (len > 1e-10f) n /= len;
                        patch_normal[pid] += n;
                        patch_size[pid]++;
                    }

                    for (unsigned int p = 0; p < patch_count; ++p) {
                        if (patch_size[p] > 0)
                            patch_normal[p] = glm::normalize(patch_normal[p]);
                    }

                    // Build patch adjacency from triangle neighbors
                    std::unordered_set<uint64_t> adj_edges;
                    adj_edges.reserve(tri_count * 2);
                    for (unsigned int i = 0; i < tri_count; ++i) {
                        int pa = tri_patch_ids[i];
                        if (pa < 0) continue;
                        for (int e = 0; e < 3; ++e) {
                            int nb = cpu_tris[i].neighbor[e];
                            if (nb < 0) continue;
                            int pb = tri_patch_ids[nb];
                            if (pb < 0 || pa == pb) continue;
                            unsigned int lo = std::min(pa, pb);
                            unsigned int hi = std::max(pa, pb);
                            adj_edges.insert((uint64_t(lo) << 32) | uint64_t(hi));
                        }
                    }

                    std::vector<int> remap(patch_count);
                    std::iota(remap.begin(), remap.end(), 0);
                    std::vector<int> psize = patch_size;
                    std::vector<glm::vec3> pnorm = patch_normal;

                    auto find = [&](int x) {
                        while (remap[x] != x) {
                            remap[x] = remap[remap[x]];
                            x = remap[x];
                        }
                        return x;
                    };

                    bool did_merge = true;
                    while (did_merge) {
                        did_merge = false;
                        for (auto& edge : adj_edges) {
                            int a = find(int(edge >> 32));
                            int b = find(int(edge & 0xFFFFFFFFu));
                            if (a == b) continue;

                            int axis_a = cpu_tris[cmds[a].firstIndex / 3u].dominant_axis;
                            int axis_b = cpu_tris[cmds[b].firstIndex / 3u].dominant_axis;
                            if (axis_a != axis_b) continue;

                            float sim = glm::dot(pnorm[a], pnorm[b]);
                            if (sim >= merge_epsilon) {
                                if (psize[a] < psize[b]) std::swap(a, b);
                                remap[b] = a;
                                pnorm[a] = glm::normalize(
                                    pnorm[a] * float(psize[a]) +
                                    pnorm[b] * float(psize[b]));
                                psize[a] += psize[b];
                                did_merge = true;
                            }
                        }
                    }

                    std::vector<int> compact(patch_count, -1);
                    int new_count = 0;
                    for (int p = 0; p < int(patch_count); ++p) {
                        int root = find(p);
                        if (compact[root] == -1)
                            compact[root] = new_count++;
                    }

                    for (unsigned int i = 0; i < tri_count; ++i) {
                        if (tri_patch_ids[i] >= 0)
                            tri_patch_ids[i] = compact[find(tri_patch_ids[i])];
                    }

                    patch_count = new_count;
                    {
                        std::vector<DrawCmd> new_cmds;
                        size_t ps = 0;
                        for (size_t i = 1; i <= tri_count; ++i) {
                            if (i == tri_count || tri_patch_ids[i] != tri_patch_ids[ps]) {
                                DrawCmd cmd;
                                cmd.count = unsigned(i - ps) * 3u;
                                cmd.instanceCount = 1;
                                cmd.firstIndex = unsigned(ps) * 3u;
                                cmd.baseVertex = 0;
                                cmd.baseInstance = 0;
                                new_cmds.push_back(cmd);
                                if (i < tri_count) ps = i;
                            }
                        }
                        cmds = std::move(new_cmds);
                    }
                }
                gllib::logf(gllib::LogLevel::info, "greedy merge: %u -> %u patches, %zu draw cmds (epsilon %.3f)",
                            pre_merge_count, patch_count, cmds.size(), merge_epsilon);


                unsigned int draw_count = unsigned(cmds.size());
                indirect_buf.data(cmds.data(), draw_count * sizeof(DrawCmd));
               count_buf.data(&draw_count, sizeof(unsigned int));


                // Per-vertex global triangle ID for occlusion lookup
                std::vector<int> tri_id(tri_count * 3, -1);
                std::vector<int> cmd_patch_id(draw_count, -1);
                for (unsigned int c = 0; c < draw_count; ++c) {
                    int first_tri = int(cmds[c].firstIndex / 3u);
                    unsigned int num_tri_p = cmds[c].count / 3u;
                    if (first_tri < int(tri_count))
                        cmd_patch_id[c] = tri_patch_ids[first_tri];
                    for (unsigned int i = 0; i < num_tri_p; ++i) {
                        int gid = first_tri + int(i);
                        for (int j = 0; j < 3; ++j)
                            tri_id[int(cmds[c].firstIndex) + int(i) * 3 + j] = gid;
                    }
                }
               tri_id_buf.data(tri_id.data(), tri_id.size() * sizeof(int));
               glVertexArrayVertexBuffer(patch_vao.handle(), 2, tri_id_buf.handle(), 0, sizeof(int));
               glVertexArrayAttribFormat(patch_vao.handle(), 2, 1, GL_INT, GL_FALSE, 0);


               // Axis ranges for sorted rendering (iterate over draw commands)
               axis_ranges.clear();
               {
                   unsigned int range_start = 0;
                   int last_axis = -1;
                   for (unsigned int c = 0; c < draw_count; ++c) {
                       unsigned int first_tri = cmds[c].firstIndex / 3u;
                       int axis = cpu_tris[first_tri].dominant_axis;
                       if (axis != last_axis) {
                           if (last_axis >= 0) {
                               unsigned int prev_tri = cmds[range_start].firstIndex / 3u;
                               axis_ranges.push_back({last_axis, cpu_tris[prev_tri].dominant_axis, range_start, c - range_start});
                           }
                           range_start = c;
                           last_axis = axis;
                       }
                   }
                   if (last_axis >= 0) {
                       unsigned int prev_tri = cmds[range_start].firstIndex / 3u;
                       axis_ranges.push_back({last_axis, cpu_tris[prev_tri].dominant_axis, range_start, draw_count - range_start});
                   }
               }


               // --- Compute per-patch metadata for atlas packing ---
               auto project_2d = [](const glm::vec3& p, int axis) -> glm::vec2 {
                   if (axis == 1 || axis == 2)  return {p.y, p.z};
                   if (axis == 4 || axis == 8)  return {p.x, p.z};
                   return {p.x, p.y};
               };
               auto height_of = [](const glm::vec3& p, int axis) -> float {
                   if (axis == 1)  return  p.x;
                   if (axis == 2)  return -p.x;
                   if (axis == 4)  return  p.y;
                   if (axis == 8)  return -p.y;
                   if (axis == 16) return  p.z;
                   return -p.z;
               };


                struct PatchCPU {
                    int dominant_axis = -1;
                    glm::vec2 proj_min{std::numeric_limits<float>::max()};
                    glm::vec2 proj_max{std::numeric_limits<float>::lowest()};
                    float height_min = std::numeric_limits<float>::max();
                    float height_max = std::numeric_limits<float>::lowest();
                };
                std::vector<PatchCPU> patch_cpu(patch_count);


                for (unsigned int i = 0; i < tri_count; ++i) {
                    int pid = tri_patch_ids[i];
                    if (pid < 0 || pid >= int(patch_count)) continue;
                    const auto& tri = cpu_tris[i];
                    int da = tri.dominant_axis;
                    auto& pc = patch_cpu[pid];
                    if (pc.dominant_axis < 0) pc.dominant_axis = da;

                    for (int v = 0; v < 3; ++v) {
                        const auto& pos = mesh.vertices()[tri.v[v]].position;
                        glm::vec3 wp = glm::vec3(model_mat * glm::vec4(pos[0], pos[1], pos[2], 1.0f));
                        glm::vec2 proj = project_2d(wp, da);
                        float h = height_of(wp, da);
                        pc.proj_min = glm::min(pc.proj_min, proj);
                        pc.proj_max = glm::max(pc.proj_max, proj);
                        pc.height_min = std::min(pc.height_min, h);
                        pc.height_max = std::max(pc.height_max, h);
                    }
                }


               // --- Atlas bin-packing ---
               glm::vec3 mesh_extent = aabb_max - aabb_min;
               float max_extent = std::max({mesh_extent.x, mesh_extent.y, mesh_extent.z});
               constexpr int atlas_target_size = 4096;
               constexpr int atlas_max_width = 8192;
               float texel_density = float(atlas_target_size) / max_extent;


               struct PatchLayout { int w, h; float ox, oy; };
               std::vector<PatchLayout> layouts(patch_count);
               for (unsigned int i = 0; i < patch_count; ++i) {
                   auto& pc = patch_cpu[i];
                   if (pc.dominant_axis < 0) { layouts[i] = {0, 0, 0, 0}; continue; }
                   glm::vec2 ext = pc.proj_max - pc.proj_min;
                   layouts[i].ox = pc.proj_min.x;
                   layouts[i].oy = pc.proj_min.y;
                   layouts[i].w = std::max(1, (int)std::ceil(ext.x * texel_density));
                   layouts[i].h = std::max(1, (int)std::ceil(ext.y * texel_density));
               }


               auto next_pow2 = [](int v) -> int { int p = 1; while (p < v) p *= 2; return p; };
               int max_patch_w = 0;
               for (auto& pl : layouts) max_patch_w = std::max(max_patch_w, pl.w);
               int min_width = std::max(256, next_pow2(max_patch_w));


               struct PackTrial { int width, height, total_texels; std::vector<PatchMetaGPU> meta; };
               std::vector<PackTrial> trials;


               for (int trial_w = min_width; trial_w <= atlas_max_width; trial_w *= 2) {
                   std::vector<int> order(patch_count);
                   std::iota(order.begin(), order.end(), 0);
                   std::sort(order.begin(), order.end(), [&](int a, int b) {
                       if (layouts[a].h != layouts[b].h) return layouts[a].h > layouts[b].h;
                       return layouts[a].w > layouts[b].w;
                   });


                   int cur_x = 0, cur_y = 0, row_h = 0;
                   std::vector<PatchMetaGPU> meta(patch_count);


                   for (int idx : order) {
                       auto& pm = meta[idx];
                       auto& pc = patch_cpu[idx];
                       auto& pl = layouts[idx];


                       pm.dominant_axis = pc.dominant_axis;
                       pm.proj_origin_x = pl.ox;
                       pm.proj_origin_y = pl.oy;
                       pm.atlas_w = pl.w;
                       pm.atlas_h = pl.h;
                       pm.height_min = pc.height_min;
                       pm.height_max = pc.height_max;


                       if (cur_x + pl.w > trial_w) { cur_x = 0; cur_y += row_h; row_h = 0; }
                       pm.atlas_x = cur_x;
                       pm.atlas_y = cur_y;
                       cur_x += pl.w;
                       row_h = std::max(row_h, pl.h);
                   }


                   int h = cur_y + row_h;
                   trials.push_back({trial_w, h, trial_w * h, std::move(meta)});
               }


               auto best = std::min_element(trials.begin(), trials.end(),
                   [](const PackTrial& a, const PackTrial& b) { return a.total_texels < b.total_texels; });


               atlas_w = best->width;
               atlas_h = best->height;
               auto& patch_meta = best->meta;


               gllib::logf(gllib::LogLevel::info, "atlas: %d x %d, %d patches, density %.1f",
                           atlas_w, atlas_h, patch_count, texel_density);


               // --- Upload GPU buffers ---
               patch_id_buf.data(tri_patch_ids.data(), tri_count * sizeof(int));
               patch_meta_buf.data(patch_meta.data(), patch_count * sizeof(PatchMetaGPU));


               // --- Create atlas textures ---
               if (height_atlas_tex)    { glDeleteTextures(1, &height_atlas_tex);    height_atlas_tex = 0; }
               if (occlusion_atlas_tex) { glDeleteTextures(1, &occlusion_atlas_tex); occlusion_atlas_tex = 0; }
               if (output_atlas_tex)    { glDeleteTextures(1, &output_atlas_tex);    output_atlas_tex = 0; }


               glCreateTextures(GL_TEXTURE_2D, 1, &height_atlas_tex);
               glTextureStorage2D(height_atlas_tex, 1, GL_R32UI, atlas_w, atlas_h);
               glTextureParameteri(height_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
               glTextureParameteri(height_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
               unsigned int clear_zero_ui = 0u;
               glClearTexImage(height_atlas_tex, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clear_zero_ui);


               glCreateTextures(GL_TEXTURE_2D, 1, &occlusion_atlas_tex);
               glTextureStorage2D(occlusion_atlas_tex, 1, GL_R8, atlas_w, atlas_h);
               glTextureParameteri(occlusion_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
               glTextureParameteri(occlusion_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
               uint8_t clear_occ = 0;
               glClearTexImage(occlusion_atlas_tex, 0, GL_RED, GL_UNSIGNED_BYTE, &clear_occ);


               glCreateTextures(GL_TEXTURE_2D, 1, &output_atlas_tex);
               glTextureStorage2D(output_atlas_tex, 1, GL_RG16F, atlas_w, atlas_h);
               glTextureParameteri(output_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
               glTextureParameteri(output_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
               glTextureParameteri(output_atlas_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
               glTextureParameteri(output_atlas_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


               // --- Pass 1: Rasterize with atomic height tracking ---
               raster_prog->use();
               tri_buf.bind_base(0);
               glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.vbo_handle());
               patch_id_buf.bind_base(2);
               patch_meta_buf.bind_base(3);
               glBindImageTexture(0, height_atlas_tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
               raster_prog->uniform1ui(loc_r_tri_cnt, GLuint(tri_count));
               raster_prog->uniform1i(loc_r_pat_cnt, GLint(patch_count));
               raster_prog->uniform1f(loc_r_density, texel_density);
               raster_prog->uniform1f(loc_r_thresh, axis_threshold);
               raster_prog->uniform_matrix4fv(loc_r_model, &model_mat[0][0]);
               glDispatchCompute(GLuint((tri_count + 255) / 256), 1, 1);
               gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


               // --- Pass 1b: Occlusion detection against finalized height atlas ---
               cpu_tri_occluded.resize(tri_count, 0);
               tri_occluded_buf.data(cpu_tri_occluded.data(), tri_count * sizeof(int));
               occ_detect_prog->use();
               tri_buf.bind_base(0);
               glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.vbo_handle());
               patch_id_buf.bind_base(2);
               patch_meta_buf.bind_base(3);
               tri_occluded_buf.bind_base(4);
               glBindImageTexture(0, height_atlas_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
               glBindImageTexture(1, occlusion_atlas_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);
               occ_detect_prog->uniform1ui(loc_o_tri_cnt, GLuint(tri_count));
               occ_detect_prog->uniform1i(loc_o_pat_cnt, GLint(patch_count));
               occ_detect_prog->uniform1f(loc_o_density, texel_density);
               occ_detect_prog->uniform1f(loc_o_thresh, axis_threshold);
               occ_detect_prog->uniform_matrix4fv(loc_o_model, &model_mat[0][0]);
               glDispatchCompute(GLuint((tri_count + 255) / 256), 1, 1);
               gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);


               glMemoryBarrier(GL_ALL_BARRIER_BITS);
               {
                   gl::Sync fence;
                   fence.client_wait(1000000000);
               }
               glGetNamedBufferSubData(tri_occluded_buf.handle(), 0, tri_count * sizeof(int), cpu_tri_occluded.data());


               // --- Pass 2: Thickness post-pass ---
               thickness_prog->use();
               glBindImageTexture(0, height_atlas_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
               glBindImageTexture(1, output_atlas_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
               glBindImageTexture(2, occlusion_atlas_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
               patch_meta_buf.bind_base(3);
               thickness_prog->uniform1i(loc_t_aw, atlas_w);
               thickness_prog->uniform1i(loc_t_ah, atlas_h);
               thickness_prog->uniform1i(loc_t_pc, GLint(patch_count));
               float global_h_min = std::numeric_limits<float>::max();
               float global_h_max = std::numeric_limits<float>::lowest();
               for (auto& pm : patch_meta) {
                   global_h_min = std::min(global_h_min, pm.height_min);
                   global_h_max = std::max(global_h_max, pm.height_max);
               }
               thickness_prog->uniform1f(loc_t_hmn, global_h_min);
               thickness_prog->uniform1f(loc_t_hrg, global_h_max - global_h_min);
               glDispatchCompute(GLuint((atlas_w + 7) / 8), GLuint((atlas_h + 7) / 8), 1);
               gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


               // --- Pass 1c: Reduce occlusion to per-patch flags ---
               cpu_occ_flags.resize(patch_count, 0);
               occ_flag_buf.data(cpu_occ_flags.data(), patch_count * sizeof(int));
               occ_reduce_prog->use();
               glBindImageTexture(0, occlusion_atlas_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
               occ_flag_buf.bind_base(0);
               patch_meta_buf.bind_base(1);
               occ_reduce_prog->uniform1i(loc_red_pc, GLint(patch_count));
               glDispatchCompute(GLuint((patch_count + 255) / 256), 1, 1);
               gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);


               glMemoryBarrier(GL_ALL_BARRIER_BITS);
               {
                   gl::Sync fence;
                   fence.client_wait(1000000000);
               }
               glGetNamedBufferSubData(occ_flag_buf.handle(), 0, patch_count * sizeof(int), cpu_occ_flags.data());


                // Build filtered indirect draw buffer (only occluded patches)
                std::vector<DrawCmd> filtered_cmds;
                for (unsigned int c = 0; c < draw_count; ++c) {
                    int pid = cmd_patch_id[c];
                    if (pid >= 0 && pid < int(patch_count) && cpu_occ_flags[pid] != 0)
                        filtered_cmds.push_back(cmds[c]);
                }
               filtered_patch_count = unsigned(filtered_cmds.size());
               filtered_indirect_buf.data(filtered_cmds.data(), filtered_patch_count * sizeof(DrawCmd));
               filtered_count_buf.data(&filtered_patch_count, sizeof(unsigned int));


                gllib::logf(gllib::LogLevel::info, "atlas rasterization + thickness complete (%u/%u patches self-occluding)",
                            filtered_patch_count, patch_count);


                // --- Write atlas textures to files ---
                if (write_textures) {
                {
                    gllib::log(gllib::LogLevel::info, "reading back atlas textures...");
                    std::vector<uint16_t> rg_half(atlas_w * atlas_h * 2);
                    glGetTextureImage(output_atlas_tex, 0, GL_RG, GL_HALF_FLOAT,
                                      atlas_w * atlas_h * 2 * sizeof(uint16_t), rg_half.data());
                    std::vector<unsigned char> rgb_u(atlas_w * atlas_h * 3);
                    for (int i = 0; i < atlas_w * atlas_h; ++i) {
                        auto to_float = [](uint16_t h) -> float {
                            unsigned int s = (h & 0x8000u) << 16;
                            unsigned int f = (h & 0x7C00u) << 13;
                            unsigned int m = h & 0x03FFu;
                            if (f == 0) {
                                if (m == 0) return reinterpret_cast<const float&>(s);
                                f = 0x00800000u;
                                while ((m & 0x0400u) == 0) { m <<= 1; f -= 0x00800000u; }
                                m &= 0x03FFu;
                            } else if (f == 0x7C00u) {
                                f = 0x7F800000u;
                            } else {
                                f += 0x38000000u;
                            }
                            unsigned int bits = s | f | (m << 13);
                            return reinterpret_cast<const float&>(bits);
                        };
                        float h = to_float(rg_half[i * 2 + 0]);
                        float t = to_float(rg_half[i * 2 + 1]);
                        float v = std::clamp(h + t * 0.3f, 0.0f, 1.0f);
                        rgb_u[i * 3 + 0] = (unsigned char)(v * 255.0f);
                        rgb_u[i * 3 + 1] = (unsigned char)(v * 255.0f);
                        rgb_u[i * 3 + 2] = (unsigned char)(v * 255.0f);
                    }
                    gllib::log(gllib::LogLevel::info, "writing atlas_output.png...");
                    stbi_write_png("atlas_output.png", atlas_w, atlas_h, 3,
                                   rgb_u.data(), atlas_w * 3);
                    gllib::log(gllib::LogLevel::info, "wrote atlas_output.png");
                }
                {
                    std::vector<unsigned char> occ_u(atlas_w * atlas_h);
                    glGetTextureImage(occlusion_atlas_tex, 0, GL_RED, GL_UNSIGNED_BYTE,
                                      atlas_w * atlas_h, occ_u.data());
                    std::vector<unsigned char> rgb_u(atlas_w * atlas_h * 3);
                    for (int i = 0; i < atlas_w * atlas_h; ++i) {
                        unsigned char v = occ_u[i];
                        rgb_u[i * 3 + 0] = v;
                        rgb_u[i * 3 + 1] = v;
                        rgb_u[i * 3 + 2] = v;
                    }
                    stbi_write_png("atlas_occlusion.png", atlas_w, atlas_h, 3,
                                   rgb_u.data(), atlas_w * 3);
                    gllib::log(gllib::LogLevel::info, "wrote atlas_occlusion.png");
                }
                } // if (write_textures)

                total_draw_count = unsigned(cmds.size());
                patches_ready = true;


               gllib::logf(gllib::LogLevel::info, "patches: %u (%.1f us GPU)",
                           patch_count, gpu_time_us);
           }


           if (patches_ready) {
               ImGui::Text("GPU time: %.1f us", gpu_time_us);
               ImGui::Text("Patches: %u", patch_count);
               ImGui::Text("Self-occluding: %u", filtered_patch_count);
               ImGui::Text("Atlas: %d x %d", atlas_w, atlas_h);
           }
           ImGui::Separator();


           ImGui::Checkbox("Debug View", &debug_view);
            ImGui::Checkbox("Occluded Only", &show_occluded_only);
            ImGui::Checkbox("Show Occlusion", &show_occlusion_3d);
            ImGui::Checkbox("Write Textures", &write_textures);
            ImGui::SliderFloat("Axis Threshold", &axis_threshold, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Merge Epsilon", &merge_epsilon, 0.0f, 1.0f, "%.3f");
           ImGui::Separator();
           ImGui::Text("Triangles: %zu", tri_count);
           ImGui::Checkbox("Indirect Draw", &use_indirect);
           ImGui::End();
       }


       renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


       render_prog->use();
       render_prog->uniform_matrix4fv(loc_vp, &cam.view_projection()[0][0]);
       render_prog->uniform_matrix4fv(loc_m, &model_mat[0][0]);
       render_prog->uniform3fv(loc_vp2, &cam.position()[0]);
       render_prog->uniform3fv(loc_ld, &light_dir[0]);
       render_prog->uniform4f(loc_c, color_val[0], color_val[1], color_val[2], 1.0f);
       render_prog->uniform1i(loc_fs, flat_shading ? 1 : 0);
       render_prog->uniform1i(loc_da, dominant_axis ? 1 : 0);
       render_prog->uniform1i(loc_dv, (debug_view && patches_ready) ? 1 : 0);
       render_prog->uniform1i(loc_pc, patch_colors ? 1 : 0);
       render_prog->uniform1i(loc_do, 0);
       render_prog->uniform1i(loc_so, (show_occlusion_3d && patches_ready) ? 1 : 0);


        if (use_indirect && patches_ready && total_draw_count > 0) {
           bool occluded_only = show_occluded_only && filtered_patch_count > 0;
           gl::Buffer& draw_indirect = occluded_only ? filtered_indirect_buf : indirect_buf;
           gl::Buffer& draw_count_buf = occluded_only ? filtered_count_buf : count_buf;
           GLsizei max_draws = occluded_only ? GLsizei(filtered_patch_count) : GLsizei(total_draw_count);
           patch_vao.bind();
           tri_occluded_buf.bind_base(4);
           glBindBuffer(GL_DRAW_INDIRECT_BUFFER, draw_indirect.handle());
           glBindBuffer(GL_PARAMETER_BUFFER, draw_count_buf.handle());
           gl::multi_draw_elements_indirect_count(
               GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 0,
               GLintptr(max_draws), sizeof(DrawCmd));
       } else {
           mesh.draw();
       }


       // --- ImGui: atlas output ---
       if (patches_ready && output_atlas_tex) {
           ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_Once);
           float display_w = float(atlas_w) * 0.2f;
           float display_h = float(atlas_h) * 0.2f;
           ImGui::SetNextWindowSize(ImVec2(display_w * 2 + 40, display_h + 80), ImGuiCond_Once);
           ImGui::Begin("Atlas Output", nullptr, ImGuiWindowFlags_NoSavedSettings);
           ImGui::Text("Atlas: %d x %d", atlas_w, atlas_h);


           ImGui::Text("Height + Thickness");
           ImGui::Image((ImTextureID)(intptr_t)output_atlas_tex, ImVec2(display_w, display_h),
                        ImVec2(0, 1), ImVec2(1, 0));
           ImGui::SameLine();
           ImGui::Text("Self-Occlusion");
           ImGui::Image((ImTextureID)(intptr_t)occlusion_atlas_tex, ImVec2(display_w, display_h),
                        ImVec2(0, 1), ImVec2(1, 0));


           ImGui::End();
       }


       gui.render();
       window.swap_buffers();
   }


   // Cleanup
   if (height_atlas_tex)    glDeleteTextures(1, &height_atlas_tex);
   if (occlusion_atlas_tex) glDeleteTextures(1, &occlusion_atlas_tex);
   if (output_atlas_tex)    glDeleteTextures(1, &output_atlas_tex);


   return EXIT_SUCCESS;
}



