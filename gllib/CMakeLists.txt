include(FetchContent)
cmake_policy(SET CMP0135 NEW)

# GLFW for windowing
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(GLFW_BUILD_EXAMPLES OFF  CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF  CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF  CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# stb_image for texture loading
FetchContent_Declare(
    stb
    URL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(stb)

# stb_image_write for screenshot export
FetchContent_Declare(
    stb_write
    URL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(stb_write)

# tinygltf for model loading
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.3
)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(imgui)

set(imgui_sources
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# GLM for math
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
set(GLMBUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLMBUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(glm)
unset(CMAKE_POLICY_VERSION_MINIMUM)

add_library(gllib
    ${CMAKE_CURRENT_SOURCE_DIR}/../glad/src/glad.c
    ${imgui_sources}
    gl/buffer.cpp
    gl/vertex_array.cpp
    gl/shader.cpp
    gl/program.cpp
    gl/texture.cpp
    gl/renderbuffer.cpp
    gl/framebuffer.cpp
    gl/state.cpp
    gl/shader_file.cpp
    gl/shader_program.cpp
    gl/program_pipeline.cpp
    gl/hot_reload.cpp
    gl/query.cpp
    gl/sampler.cpp
    gl/sync.cpp
    gl/debug.cpp
    gl/atomic_counter.cpp
    gfx/window.cpp
    gfx/texture.cpp
    gfx/mesh.cpp
    gfx/material.cpp
    gfx/renderer.cpp
    gfx/camera.cpp
    gfx/cubemap.cpp
    gfx/coverage_atlas.cpp
    gfx/gpu_pipeline.cpp
    gfx/model.cpp
    gfx/bindless.cpp
    gfx/light.cpp
    gfx/gpu_particle_system.cpp
    gfx/post_processing.cpp
    gfx/renderpass.cpp
    gfx/scene.cpp
    gfx/skybox.cpp
    gfx/ibl_probe.cpp
    gfx/shadow_map.cpp
    gfx/pbr_material.cpp
    gfx/depth_pyramid.cpp
    gfx/debug_draw.cpp
    gfx/gpu_timer.cpp
    gfx/screenshot.cpp
    gfx/gbuffer.cpp
    gfx/skeleton.cpp
    gfx/animation.cpp
    gfx/imgui_overlay.cpp
    gfx/fsr2.cpp
    gllib/log.cpp
)

target_include_directories(gllib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../glad/include
        ${glm_SOURCE_DIR}
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${stb_write_SOURCE_DIR}
    PRIVATE
        ${stb_SOURCE_DIR}
        ${tinygltf_SOURCE_DIR}
)

target_link_libraries(gllib
    PUBLIC
        OpenGL::GL
        glfw
)

# FidelityFX FSR2 — CPU API only (backend implemented in gfx/fsr2_backend_gl.cpp)
FetchContent_Declare(
    ffx_fsr2
    GIT_REPOSITORY /tmp/opencode/FidelityFX-FSR2-OpenGL
    GIT_TAG dd62b03
    SOURCE_SUBDIR src/ffx-fsr2-api
    GIT_SUBMODULES ""
)
FetchContent_GetProperties(ffx_fsr2)
if(NOT ffx_fsr2_POPULATED)
    FetchContent_Populate(ffx_fsr2)
endif()

target_compile_definitions(gllib
    PUBLIC
        GLLIB_GL_NAMESPACE=1
        FFX_GCC=1
)
target_include_directories(gllib
    PUBLIC
        ${ffx_fsr2_SOURCE_DIR}/src/ffx-fsr2-api
)
target_sources(gllib PRIVATE
    gfx/fsr2_cpu_api.cpp
    gfx/fsr2_backend_gl.cpp
    ${ffx_fsr2_SOURCE_DIR}/src/ffx-fsr2-api/ffx_assert.cpp
)

# ffx_types.h uses size_t without including <cstddef>; force-include compat header
set_source_files_properties(
    ${ffx_fsr2_SOURCE_DIR}/src/ffx-fsr2-api/ffx_assert.cpp
    PROPERTIES COMPILE_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/gfx/ffx_compat.h"
)

target_compile_definitions(gllib PRIVATE
    FSR2_SHADERS_DIR="${ffx_fsr2_SOURCE_DIR}/src/ffx-fsr2-api/shaders"
)
