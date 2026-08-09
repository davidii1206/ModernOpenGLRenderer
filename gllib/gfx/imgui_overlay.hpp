#pragma once

namespace gfx {

class Window;

class ImGuiOverlay {
public:
    ImGuiOverlay() = default;
    ~ImGuiOverlay();

    ImGuiOverlay(const ImGuiOverlay&) = delete;
    ImGuiOverlay& operator=(const ImGuiOverlay&) = delete;

    bool init(const Window& window);
    void shutdown();
    bool active() const { return active_; }

    void begin_frame();
    void render();
    void end_frame() {} // convenience alias kept for compatibility

    bool wants_mouse() const;
    bool wants_keyboard() const;

private:
    bool active_ = false;
};

} // namespace gfx
