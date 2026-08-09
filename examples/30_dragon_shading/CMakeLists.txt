add_executable(30_dragon_shading main.cpp)
target_link_libraries(30_dragon_shading PRIVATE gllib)
target_compile_features(30_dragon_shading PRIVATE cxx_std_20)
add_custom_command(TARGET 30_dragon_shading POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Dragon.glb $<TARGET_FILE_DIR:30_dragon_shading>/Stanford_Dragon.glb
)
