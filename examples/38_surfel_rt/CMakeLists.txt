add_executable(38_surfel_rt main.cpp)
target_link_libraries(38_surfel_rt PRIVATE gllib)
target_compile_features(38_surfel_rt PRIVATE cxx_std_20)

add_custom_command(TARGET 38_surfel_rt POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:38_surfel_rt>/shaders
)
add_custom_command(TARGET 38_surfel_rt POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_SOURCE_DIR}/data/CornellBoxMirrorBox.glb
          $<TARGET_FILE_DIR:38_surfel_rt>
)
