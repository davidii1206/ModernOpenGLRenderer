add_executable(36_micro_gi main.cpp)
target_link_libraries(36_micro_gi PRIVATE gllib)
target_compile_features(36_micro_gi PRIVATE cxx_std_20)
add_custom_command(TARGET 36_micro_gi POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb $<TARGET_FILE_DIR:36_micro_gi>/CornellBoxOriginal.glb
)
