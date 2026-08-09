add_executable(29_mdc_animation main.cpp)
target_link_libraries(29_mdc_animation PRIVATE gllib)
target_compile_features(29_mdc_animation PRIVATE cxx_std_20)
add_custom_command(TARGET 29_mdc_animation POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/BrainStem.glb $<TARGET_FILE_DIR:29_mdc_animation>/BrainStem.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:29_mdc_animation>/Stanford_Dragon.glb
)
