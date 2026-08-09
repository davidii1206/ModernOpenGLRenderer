add_executable(16_bindless main.cpp)
target_link_libraries(16_bindless PRIVATE gllib)
target_compile_features(16_bindless PRIVATE cxx_std_20)
