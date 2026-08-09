#include "imgui_overlay.hpp"
#include "window.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

namespace gfx {

ImGuiOverlay::~ImGuiOverlay() {
    shutdown();
}

bool ImGuiOverlay::init(const Window& window) {
    if (active_) return true;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;

    GLFWwindow* handle = static_cast<GLFWwindow*>(window.native_handle());
    if (!handle) return false;

    if (!ImGui_ImplGlfw_InitForOpenGL(handle, true))
        return false;
    if (!ImGui_ImplOpenGL3_Init("#version 460")) {
        ImGui_ImplGlfw_Shutdown();
        return false;
    }

    active_ = true;
    return true;
}

void ImGuiOverlay::shutdown() {
    if (!active_) return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    active_ = false;
}

void ImGuiOverlay::begin_frame() {
    if (!active_) return;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiOverlay::render() {
    if (!active_) return;
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool ImGuiOverlay::wants_mouse() const {
    if (!active_) return false;
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiOverlay::wants_keyboard() const {
    if (!active_) return false;
    return ImGui::GetIO().WantCaptureKeyboard;
}

} // namespace gfx
