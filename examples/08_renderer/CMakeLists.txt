add_executable(08_renderer main.cpp)
target_link_libraries(08_renderer PRIVATE gllib)
target_compile_features(08_renderer PRIVATE cxx_std_20)
