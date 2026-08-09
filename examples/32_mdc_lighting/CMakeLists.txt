add_executable(32_mdc_lighting main.cpp)
target_link_libraries(32_mdc_lighting PRIVATE gllib)
target_compile_features(32_mdc_lighting PRIVATE cxx_std_20)
add_custom_command(TARGET 32_mdc_lighting POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb $<TARGET_FILE_DIR:32_mdc_lighting>/CornellBoxOriginal.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb $<TARGET_FILE_DIR:32_mdc_lighting>/Stanford_Bunny.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:32_mdc_lighting>/Stanford_Dragon.glb
)
