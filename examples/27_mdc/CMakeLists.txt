add_executable(27_mdc main.cpp)
target_link_libraries(27_mdc PRIVATE gllib)
target_compile_features(27_mdc PRIVATE cxx_std_20)
add_custom_command(TARGET 27_mdc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:27_mdc>/Stanford_Dragon.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb $<TARGET_FILE_DIR:27_mdc>/Stanford_Bunny.glb
)
