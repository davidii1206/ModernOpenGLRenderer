add_executable(37_emissive_gi main.cpp)
target_link_libraries(37_emissive_gi PRIVATE gllib)
target_compile_features(37_emissive_gi PRIVATE cxx_std_20)

add_custom_command(TARGET 37_emissive_gi POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:37_emissive_gi>/shaders
)
add_custom_command(TARGET 37_emissive_gi POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb
          ${CMAKE_SOURCE_DIR}/data/sponza.glb
          $<TARGET_FILE_DIR:37_emissive_gi>
)
