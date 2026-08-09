# gllib — Modern OpenGL 4.6 C++ Library

A two-layer C++ library for modern OpenGL 4.6, designed for high-performance GPU-driven
rendering, compute shaders, and fast prototyping.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                    gfx::                          │
│   High-level API                                  │
│   (mesh, model, material, texture, scene, GPU     │
│    pipeline, ImGui overlay, FSR upscaling)        │
├───────────────────────────────────────────────────┤
│                    gl::                            │
│   Low-level API                                   │
│   (buffer, VAO, shader, program, texture, FBO,    │
│    queries, sync, compute dispatch, hot-reload)   │
├───────────────────────────────────────────────────┤
│              glad (OpenGL 4.6 core + extensions)  │
└───────────────────────────────────────────────────┘
```

Both APIs live under `gllib/`. The `gl::` and `gfx::` namespaces are distinct so both
can be used in the same project.

## Requirements

| Requirement | Minimum |
|-------------|---------|
| C++ standard | C++20 |
| Compiler | GCC 11+, Clang 14+, MSVC 2022+ |
| OpenGL | 4.6 core profile |
| Platform | Linux (X11/Wayland), Windows |
| CMake | 3.16+ |

macOS is not supported — Apple's OpenGL implementation is stuck at 4.1 and does not provide the 4.6 features this library depends on (DSA, compute shaders, indirect draws, ARB_bindless_texture).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Examples are built by default (`-DGLLIB_BUILD_EXAMPLES=OFF` to disable).
When Doxygen is found, `cmake --build build --target docs` generates API docs.

**Dependencies** (all pulled via CMake `FetchContent`):
- glad2 (OpenGL 4.6 loader)
- GLFW 3.x (windowing, context, input)
- GLM (math)
- tinygltf (model loading)
- Dear ImGui v1.91+ (debug UI)

## Using gllib in your project

### As a CMake subdirectory

```cmake
# Copy or submodule gllib into your project, then:
add_subdirectory(path/to/gllib)
target_link_libraries(your_target PRIVATE gllib)
```

### Via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(gllib
    GIT_REPOSITORY https://github.com/your/repo.git
    GIT_TAG        main-or-tag
)
FetchContent_MakeAvailable(gllib)
target_link_libraries(your_target PRIVATE gllib)
```

The `gllib` target automatically provides all include paths and links OpenGL + GLFW. No additional setup needed.

## License

MIT. See [LICENSE](LICENSE) for details.

## Examples

| # | Example | Highlights |
|---|---------|------------|
| 01 | Hello triangle | Minimal OpenGL triangle, `gl::Buffer`, `gl::Shader`, `gl::Program` |
| 02 | Clear screen | `gfx::Window`, `gl::clear`, `gl::clear_color` |
| 03 | Buffer test | Interleaved vertex attributes (pos + color), `offsetof` stride |
| 04 | Pure glib | Zero raw GL calls, wrapper-only triangle |
| 05 | Shader file | `gl::program_from_files`, external `vert.glsl`/`frag.glsl` |
| 06 | Texture | Procedural checkerboard via `gfx::Texture`, mipmaps, indexed quad |
| 07 | Mesh + Material | `gfx::Mesh` + `gfx::Material` abstraction, textured cube, UV scroll |
| 08 | Renderer | `gfx::Renderer` draw call management, animated rotating cube |
| 09 | Camera | Interactive orbit controls (arrow keys + W/S), perspective projection |
| 10 | Model | `gfx::Model` glTF/GLB loading with mouse-drag orbit + scroll zoom |
| 11 | Sampler | `gl::Sampler` separate sampler objects, 4 filtering/wrap modes |
| 12 | RenderPass | `gfx::RenderPass` offscreen RTT, FBO blit, scissor test, dual viewport |
| 13 | Scene graph | `gfx::Scene`/`gfx::Node` transform hierarchy, orbiting children |
| 14 | Compute | `gl::dispatch_compute` + `gl::Sync`, compute-generated particle points |
| 15 | LightBuffer | `gfx::LightBuffer` UBO, directional + animated point lights, Blinn-Phong |
| 16 | Bindless | `gfx::BindlessManager`, `ARB_bindless_texture` sampler from `uvec2` handles |
| 17 | Query | `gl::Query` timer (`GL_TIME_ELAPSED`) + occlusion (`GL_ANY_SAMPLES_PASSED`) |
| 18 | Indirect draw | `gl::multi_draw_elements_indirect` with `gl_DrawID` and SSBO per-draw offsets |
| 19 | GPU particles | `gfx::GpuParticleSystem`, fully GPU-driven emit/simulate/count, additive blend |
| 20 | Environment | `gfx::Cubemap` + `gfx::Skybox`, procedural cubemap, reflective Fresnel cube |
| 21 | Post-processing | `gfx::PostProcessing`, HDR + bloom + ACES tonemapping, 6 overdriven cubes |
| **22** | **GPU pipeline** | **Frustum + HiZ occlusion cull, LOD, indirect multi-draw, FSR 1, ImGui stats** |
| **23** | **ImGui demo** | **Standalone `gfx::ImGuiOverlay` with widgets over a colored triangle** |
| **24** | **Debug draw** | **`gfx::DebugDraw` immediate-mode line/box/sphere/frustum/axis overlay** |
| **25** | **PBR / IBL** | **Cook-Torrance BRDF, IBL split-sum (`IBLProbe`), hard-shadow directional light (`ShadowMap`), procedural HDR skybox (`Skybox`), Sponza model, ACES tonemapping** |
| **26** | **Deferred PBR** | **Geometry pass → G-buffer (`gfx::GBuffer`) → lighting pass; fullscreen quad PBR with IBL + hard shadow, ImGui G-buffer viewer (albedo/normal/depth)** |
| **27** | **MDC — convex clustering** | **Edge-list extraction + SSBO, GPU edge classification (signed dihedral angle), forward PBR/IBL, debug overlay with reflex edge stats** |
| **28** | **Animation** | **Skinned character, bone matrices, PBR/IBL, ImGui controls** |
| **29** | **MDC + Animation** | **Skinned character (MDC every frame) + static dragon, dirty-bone gating** |
| **30** | **Dragon shading** | **Stanford Dragon: adjacency-based patch system, GPU patch detection (prefix sum), indirect draw, self-occlusion debug** |
| **31** | **Patch atlas** | **Mesh → occlusion-free patch atlas pipeline: convex clustering, atlas packing, BVH, adaptive MIP4 sparse mip chains (see below)** |

Example 31 (mesh → patch-atlas pipeline) is a thin driver over the CPU-only
`gfx::CoverageAtlas` API (`gllib/gfx/coverage_atlas.hpp`): it loads a model, calls
`build()` + `write_files()`, then renders the patches with OpenGL multi-draw indirect.
It writes `atlas_depth.bin`, `atlas_thickness.bin`, `atlas_uv.bin` in the
self-describing **MIP4** sparse mip-chain format —
see [docs/mip4.md](docs/mip4.md). `tools/mip4_to_bmp.py` renders a chain to a BMP for
visual inspection.

## Feature Overview

### `gl::` — Low-level RAII wrappers

| Object | File | Key features |
|--------|------|-------------|
| `Buffer` | `gl/buffer.hpp` | Generic GPU buffer, immutable storage, mapping, `storage()` with explicit flags, `map_range()` for persistent/coherent access |
| `VertexArray` | `gl/vertex_array.hpp` | DSA-style VAO setup with separate attrib format |
| `Shader` | `gl/shader.hpp` | Compile from source, SPIR-V loading |
| `Program` | `gl/program.hpp` | Link, attach, uniform helpers |
| `ProgramPipeline` | `gl/program_pipeline.hpp` | Separable shader programs |
| `Texture` | `gl/texture.hpp` | 1D/2D/3D, cubemaps, immutable storage, bindless |
| `Framebuffer` | `gl/framebuffer.hpp` | FBO with DSA attachment, blit, completeness check |
| `Sampler` | `gl/sampler.hpp` | Separate sampler objects, bindless compatible |
| `Renderbuffer` | `gl/renderbuffer.hpp` | Renderbuffer storage |
| `Sync` | `gl/sync.hpp` | `glFenceSync` RAII wrapper |
| `Query` | `gl/query.hpp` | Timer queries (`GL_TIME_ELAPSED`) + occlusion (`GL_ANY_SAMPLES_PASSED`) |
| `AtomicCounter` | `gl/atomic_counter.hpp` | RAII wrapper for `GL_ATOMIC_COUNTER_BUFFER`, reset via `glClearNamedBufferData` |
| `ShaderStorageBuffer` | `gl/ssbo.hpp` | Typed SSBO helper (header-only) |
| `UniformBuffer` | `gl/ubo.hpp` | Typed UBO helper (header-only) |
| `HotReloadProgram` | `gl/hot_reload.hpp` | File-watch + recompile on modification |
| `shader_include` | `gl/shader_include.hpp` | `#include` resolution for GLSL (no driver extension needed) |
| `DebugOutput` | `gl/debug.hpp` | `glDebugMessageCallback` wrapper, `glObjectLabel` helpers, `GL_CHECK` macro |

### `gfx::` — High-level API

| Class | File | Description |
|-------|------|-------------|
| `Window` | `gfx/window.hpp` | GLFW wrapper — input, vsync, scroll, framebuffer resize, debug context flag |
| `Renderer` | `gfx/renderer.hpp` | Clear, viewport, state sorting |
| `Camera` | `gfx/camera.hpp` | Perspective/ortho, orbit controls, view-projection |
| `Texture` | `gfx/texture.hpp` | File I/O (stb_image), GPU upload, bindless support |
| `Mesh` | `gfx/mesh.hpp` | Vertex + index buffers, VAO, submeshes |
| `Material` | `gfx/material.hpp` | Program + uniform management |
| `Model` | `gfx/model.hpp` | GLB/glTF loader (tinygltf), LOD chain detection, bounding spheres |
| `Scene` / `Node` | `gfx/scene.hpp` | Scene graph with transform hierarchy |
| `LightBuffer` | `gfx/light.hpp` | UBO-backed dir/point/spot lights |
| `BindlessManager` | `gfx/bindless.hpp` | `GL_ARB_bindless_texture` state tracking |
| `RenderPass` | `gfx/renderpass.hpp` | FBO + clear + draw config |
| `GpuPipeline` | `gfx/gpu_pipeline.hpp` | **Full GPU-driven pipeline** (see below) |
| `ImGuiOverlay` | `gfx/imgui_overlay.hpp` | **Dear ImGui init/newframe/render lifecycle** |
| `DepthPyramid` | `gfx/depth_pyramid.hpp` | **Hi-Z max-reduction mip chain for occlusion culling** |
| `CoverageAtlas` | `gfx/coverage_atlas.hpp` | **CPU-only mesh → occlusion-free patch atlas: clustering, packing, BVH, MIP4 mip chains (docs/mip4.md)** |
| `DebugDraw` | `gfx/debug_draw.hpp` | **Immediate-mode line/box/sphere/frustum/axis drawing** |
| `GpuTimer` | `gfx/gpu_timer.hpp` | **GPU timer convenience — ping-pong query pair, automated begin/end/poll/reuse** |
| `screenshot` | `gfx/screenshot.hpp` | **Save current framebuffer to PNG via stb_image_write** |
| `AssetCache` | `gfx/asset_cache.hpp` | **Path-keyed resource cache** |
| `FsrHelpers` | `gfx/fsr.hpp` | **FSR 1 EASU/RCAS constant computation** |

### GPU-Driven Pipeline (`gfx::GpuPipeline`)

The marquee feature — a fully GPU-bound rendering pipeline with zero CPU–GPU sync:

1. **Frustum culling** — compute shader tests bounding spheres against view frustum
2. **Hi-Z occlusion culling** — max-reduction depth pyramid from previous frame; project
   bounding sphere, sample at appropriate mip, discard occluded
3. **LOD selection** — projected pixel size from bounding sphere + projection → per-LOD
   visible-ID packing via atomic counters
4. **Indirect multi-draw** — packed visible IDs → `DrawElementsIndirectCommand` via second
   compute pass → single `glMultiDrawElementsIndirectCount` with merged VAO
5. **FSR 1 upscaling** — EASU (edge-adaptive spatial upsampling) + RCAS (contrast-adaptive
   sharpening) as two compute passes; render at reduced resolution, upscale to display.
   (FSR 2 is not planned — see below.)

**Key design decisions:**
- Packed visible-ID arrays sidestep the radeonsi `gl_InstanceID`/`baseInstance` bug
- `gl_DrawID` in vertex shader selects LOD mesh data from merged VAO
- Hi-Z from previous frame — no extra geometry pass needed
- All stages timed via `gl::Query` and displayed in the ImGui stats overlay

### FSR 1 Integration

- **Downscaling**: configurable `render_scale` (0.25–1.5), FBOs at reduced resolution
- **EASU compute**: 8×8 workgroups, `textureGather`-based 12-tap Lanczos upscale
- **RCAS compute**: 3×3 contrast-adaptive sharpening
- **Quality presets**: Low (0.33) / Medium (0.5) / High (0.75) / Ultra (1.0) with
  per-tier sharpness values, selectable via ImGui combo box
- All uniforms packed as `uvec4` (FSR convention — raw float bits in uint)

## What's Still Missing

### Critical Gaps

| Gap | Details |
|-----|---------|
| **Testing** | No test infrastructure at all. Every component is exercised only implicitly via examples. No regression safety. |
| **Error handling & debugging** | ✅ `gl::enable_debug_output()`, `gl::set_debug_callback()`, `glObjectLabel` wrappers, and `GL_CHECK` macro added. Use `WindowDesc::debug = true` to enable the debug context. The debug callback prints to stderr and aborts on high-severity errors. |
| **GL capability query** | No API to query device features/limits (`GL_MAX_SHADER_STORAGE_BLOCK_SIZE`, `GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS`, extension support). |
| **Persistent mapped buffers** | ✅ `gl::Buffer::storage()` with explicit flags + `map_range()` added. Use `GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT` for zero-copy streaming. |
| **Pipeline / program cache** | No mechanism to cache compiled SPIR-V or binary programs to disk. Hot-reload exists but every run recompiles from source. |
| **Screenshot / readback utility** | ✅ `gfx::screenshot("frame.png")` added. Uses `glReadPixels` + `stb_image_write`, flips rows, captures current viewport by default. |
| **GPU timer convenience** | ✅ `gfx::GpuTimer` added — ping-pong query pair, scoped begin/end, `available()` + `elapsed_ns()`/`elapsed_ms()`. |
| **Threading / async loading** | Model loading blocks the main thread. No async resource loading API. |
| **Code quality tooling** | No `.clang-tidy`, `.clang-format`, `.editorconfig`. No CI. |

### PBR / IBL

Delivered in example 25: Cook-Torrance BRDF, IBL split-sum (`IBLProbe`), hard-shadow directional light (`ShadowMap`), procedural HDR skybox (`Skybox`), ACES filmic tonemapping, alpha handling, and ambient hemisphere fill light. The example uses composable library classes (`PBRMaterial`, `IBLProbe`, `ShadowMap`, `Skybox`) rather than inline code, with an ImGui overlay for shadow-map debugging.

### Remaining Features

#### High Impact

| Feature | Rationale |
|---------|-----------|
| **Cascaded Shadow Maps (CSM)** | 3–4 split cascades. Natural successor to basic shadow maps (which don't exist as an example yet). Pairs well with `GpuPipeline`. |
| **`gl::Query` — occlusion + pipeline stats** | `GL_ANY_SAMPLES_PASSED` exists in example 17 but isn't surfaced in the high-level wrapper. `ARB_pipeline_statistics_query` missing entirely. |

#### Medium Impact

| Feature | Rationale |
|---------|-----------|
| **Screen-Space Ambient Occlusion (SSAO)** | HBAO+ or GTAO as a `gfx::RenderPass` post-process. Natural companion to PBR/IBL. |
| **Shader permutation / variant system** | No way to compile variants with different `#define` combinations. A `gl::ProgramVariants` would round out the hot-reload + `#include` infrastructure. |

| **`gfx::Decal` system** | Deferred decals projected onto geometry via depth reconstruction. Natural extension of the post-processing pipeline. |
| **Multi-Sample Anti-Aliasing (MSAA)** | No support for multisampled FBOs or `gl::Texture` with `GL_TEXTURE_2D_MULTISAMPLE`. Some users may prefer MSAA over TAA or FSR. |
| **Animation (skeletal)** | glTF already carries skeleton + animation data (tinygltf parses it) but `gfx::Model` doesn't surface it. Needs `gfx::Skeleton`, `gfx::AnimationClip`, `gfx::AnimatedModel` — keyframe interpolation, matrix palette SSBO, GPU skinning in the vertex shader. Animation state machines/blend trees belong in a game engine layer above gllib. |

#### Lower Priority

| Feature | Rationale |
|---------|-----------|
| **Sparse textures** | `ARB_sparse_texture` (virtual texturing) is the other major 4.6 texture feature beyond bindless. |
| **Single-header distribution** | A script that flattens `gllib/` into `gllib_single.hpp` for drop-in use. |
| **`gfx::Terrain`** | Heightmap + tessellation shader pass. Showcases LOD + GPU culling on a large scene. |

### FSR 2

FSR 2 is not planned — it relies on temporal history (motion vectors, TAA) that is incompatible with OpenGL's lack of standardized GPU-driven scheduling and low adoption of `GL_ARB_sparse_texture2`. The library focuses on native OpenGL 4.6 features instead.

### Rendering Paths

The library is designed to support multiple rendering paths without tying users into a single approach:

| Path | Status | Use case |
|------|--------|----------|
| **Forward** | ✅ Works now via `gfx::Material` + `gfx::LightBuffer` (example 15) | Simple scenes, transparency, MSAA |
| **Deferred** | ✅ Done — `gfx::GBuffer` (example 26), fullscreen lighting pass with PBR + IBL + hard shadow | Many lights, complex materials, screen-space effects |
| **Forward+** | ❌ Possible — clustered light culling via compute, then forward shade | Large light counts with transparency support |

**Forward** is the current path. `gfx::Material` manages a program + uniforms without baking in any shading model, so users can write their own Blinn-Phong, PBR, toon, or anything else. Works with MSAA trivially since there's no G-buffer to resolve.

**Deferred** (example 26) is the high-performance path. A `gfx::GBuffer` FBO with a documented layout (RT0=RGB8_SRGB albedo, RT1=RGBA16F normal.xy+roughness+metallic, RT2=RGBA16F emissive+AO, depth=GL_DEPTH_COMPONENT24) serves as the contract. Users write their own lighting GLSL that binds the G-buffer textures by name — no virtual dispatch, no inheritance. The existing `GpuPipeline` writes to MRT with minimal changes; culling, LOD, and indirect draw stay identical. SSAO becomes a standalone compute pass that reads depth+normal and writes AO into the G-buffer before lighting.

**Forward+** is left as an open possibility. Anyone can implement clustered/tiled light culling as a compute shader using `gl::dispatch_compute` and `gl::ShaderStorageBuffer`, then render forward with the light list in an SSBO. No library changes needed — `gfx::LightBuffer` and `gfx::Material` already compose.

All three paths share the same `gfx::Camera`, `gfx::LightBuffer`, `gfx::Model`, `gl::Query`, and ImGui infrastructure. They differ only in how geometry meets lighting.

### Anti-Aliasing

| Method | Status | Notes |
|--------|--------|-------|
| **MSAA** | ❌ Not supported | `gl::Texture` has no `GL_TEXTURE_2D_MULTISAMPLE` path. Forward rendering benefits most; deferred needs custom resolve. |
| **FSR 1** | ✅ Done | Spatial upscaler (EASU + RCAS). Reduces aliasing by rendering at higher resolution, but not a true AA solution. |

Forward rendering users who prefer MSAA over post-process AA should be able to enable it once `gl::Texture` gains multisample support. When deferred is added, MSAA will require a custom resolve in the lighting pass.

### Design Philosophy on Lighting / Shadows

`gfx::Material` is intentionally generic — it manages a program and its uniforms without baking in any particular shading model. This leaves the door open for anyone to implement their own lighting solution using `gfx::` primitives without fighting a predefined PBR pipeline. When a `gfx::PBRMaterial` or shadow-mapping utility is added, it should compose with `gfx::Material` rather than replace it.

Similarly, `gfx::GBuffer`, `gfx::LightingPass`, and any shadow-mapping utility should be opt-in components, not baked into `GpuPipeline` or `gfx::Renderer`. The library provides building blocks; the user composes them.
