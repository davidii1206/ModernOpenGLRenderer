# gllib — Modern C++ OpenGL 4.6 Wrapper Library {#mainpage}

## Layers

| Namespace | Description |
|-----------|-------------|
| `gl::` | Low-level RAII wrappers — Buffer, Texture, Shader, Program, Sampler, Sync, FBO, Query, SSBO, UBO, compute dispatch, hot-reload |
| `gfx::` | High-level fast-prototyping API — Window, Camera, Mesh, Material, Model (GLB/glTF), Scene, Light, RenderPass, BindlessManager, GpuPipeline, DepthPyramid, ImGuiOverlay, FSR helpers |
| `gllib::` | Logging helpers |

## Quick Start

```cpp
#include <gfx/gfx.hpp>

int main() {
    gfx::Window window({ .width = 800, .height = 600, .title = "Hello" });
    gfx::Renderer renderer;

    while (!window.should_close()) {
        renderer.clear();
        window.swap_buffers();
    }
}
```

## Examples

See `examples/` for 23 standalone examples (01–23):

| # | Topic | Highlights |
|---|-------|------------|
| 01–14 | Fundamentals | Clear → compute shader, progressive feature introduction |
| 15–21 | Advanced rendering | PBR, bindless, particles, shadow maps, environment maps, post-processing |
| **22** | **GPU-driven** | **Frustum cull + HiZ occlusion + LOD on GPU, indirect multi-draw, FSR 1, ImGui stats** |
| **23** | **ImGui demo** | **Standalone Dear ImGui integration test** |

## GPU-Driven Pipeline

Example 22 (`22_gpu_driven`) is the project's flagship demo — a fully GPU-bound
rendering pipeline:

```
         ┌─────────────┐
         │  Hi-Z Build │  ← previous frame's depth → max-reduction mip chain
         └──────┬──────┘
                ↓
         ┌─────────────┐
         │  Frustum    │  ← bounding sphere × view-proj → discard outside
         │  + HiZ Cull │  ← sphere projected, HiZ sampled at matching LOD
         └──────┬──────┘
                ↓
         ┌─────────────┐
         │  LOD Select │  ← projected pixel size → per-LOD atomic counters
         │  + Pack IDs │
         └──────┬──────┘
                ↓
         ┌─────────────┐
         │  Indirect   │  ← packed IDs → DrawElementsIndirectCommand per slot
         │  Draw       │  ← glMultiDrawElementsIndirectCount (single call)
         └──────┬──────┘
                ↓
         ┌─────────────┐
         │  FSR 1      │  ← EASU upscale + RCAS sharpening (compute passes)
         │  Upscale    │
         └──────┬──────┘
                ↓
            ┌────────┐
            │ Screen │
            └────────┘
```

All stages run entirely on the GPU with no CPU readback (except optional debug
readback toggled by `R` key). Per-frame timings for each stage are displayed
via the ImGui overlay.
