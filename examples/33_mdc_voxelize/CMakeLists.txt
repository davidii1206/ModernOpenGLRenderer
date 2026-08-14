add_executable(33_mdc_voxelize main.cpp)
target_link_libraries(33_mdc_voxelize PRIVATE gllib)
target_compile_features(33_mdc_voxelize PRIVATE cxx_std_20)
add_custom_command(TARGET 33_mdc_voxelize POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb $<TARGET_FILE_DIR:33_mdc_voxelize>/CornellBoxOriginal.glb
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb    $<TARGET_FILE_DIR:33_mdc_voxelize>/Stanford_Bunny.glb
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb   $<TARGET_FILE_DIR:33_mdc_voxelize>/Stanford_Dragon.glb
)
