add_executable(10_model main.cpp)
target_link_libraries(10_model PRIVATE gllib)
target_compile_features(10_model PRIVATE cxx_std_20)
add_custom_command(TARGET 10_model POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/cube.gltf $<TARGET_FILE_DIR:10_model>/cube.gltf
)
