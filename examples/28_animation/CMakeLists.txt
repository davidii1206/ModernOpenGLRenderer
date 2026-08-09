add_executable(28_animation main.cpp)
target_link_libraries(28_animation PRIVATE gllib)
target_compile_features(28_animation PRIVATE cxx_std_20)
add_custom_command(TARGET 28_animation POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/BrainStem.glb $<TARGET_FILE_DIR:28_animation>/BrainStem.glb
)
