add_executable(35_ssrc main.cpp)
target_link_libraries(35_ssrc PRIVATE gllib)
target_compile_features(35_ssrc PRIVATE cxx_std_20)
add_custom_command(TARGET 35_ssrc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb $<TARGET_FILE_DIR:35_ssrc>/CornellBoxOriginal.glb
)
