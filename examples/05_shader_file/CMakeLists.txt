add_executable(05_shader_file main.cpp)
target_link_libraries(05_shader_file PRIVATE gllib)
target_compile_features(05_shader_file PRIVATE cxx_std_20)

add_custom_command(TARGET 05_shader_file POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/vert.glsl
        ${CMAKE_CURRENT_SOURCE_DIR}/frag.glsl
        $<TARGET_FILE_DIR:05_shader_file>
    COMMENT "Copying shader files to executable directory"
)
