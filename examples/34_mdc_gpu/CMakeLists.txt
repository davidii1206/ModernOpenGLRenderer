add_executable(34_mdc_gpu main.cpp)
target_link_libraries(34_mdc_gpu PRIVATE gllib)
target_compile_features(34_mdc_gpu PRIVATE cxx_std_20)
add_custom_command(TARGET 34_mdc_gpu POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:34_mdc_gpu>/Stanford_Dragon.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb $<TARGET_FILE_DIR:34_mdc_gpu>/CornellBoxOriginal.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb $<TARGET_FILE_DIR:34_mdc_gpu>/Stanford_Bunny.glb
)
