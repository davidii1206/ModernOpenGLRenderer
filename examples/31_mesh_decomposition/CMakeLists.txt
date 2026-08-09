add_executable(31_mesh_decomposition main.cpp)
target_link_libraries(31_mesh_decomposition PRIVATE gllib)
target_compile_features(31_mesh_decomposition PRIVATE cxx_std_20)
add_custom_command(TARGET 31_mesh_decomposition POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:31_mesh_decomposition>/Stanford_Dragon.glb
)
