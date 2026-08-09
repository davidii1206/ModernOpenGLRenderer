add_executable(04_pure_gl main.cpp)
target_link_libraries(04_pure_gl PRIVATE gllib)
target_compile_features(04_pure_gl PRIVATE cxx_std_20)
