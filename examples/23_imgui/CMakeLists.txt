add_executable(23_imgui main.cpp)
target_link_libraries(23_imgui PRIVATE gllib)
target_compile_features(23_imgui PRIVATE cxx_std_20)
