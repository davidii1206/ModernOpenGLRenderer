add_executable(22_gpu_driven main.cpp)
target_link_libraries(22_gpu_driven PRIVATE gllib)
target_compile_features(22_gpu_driven PRIVATE cxx_std_20)
add_custom_command(TARGET 22_gpu_driven POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb $<TARGET_FILE_DIR:22_gpu_driven>/Stanford_Bunny.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/Rocks.glb $<TARGET_FILE_DIR:22_gpu_driven>/Rocks.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/DifferentRock.glb $<TARGET_FILE_DIR:22_gpu_driven>/DifferentRock.glb
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/vert.glsl $<TARGET_FILE_DIR:22_gpu_driven>/vert.glsl
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/frag.glsl $<TARGET_FILE_DIR:22_gpu_driven>/frag.glsl
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/fsr_easu.glsl $<TARGET_FILE_DIR:22_gpu_driven>/fsr_easu.glsl
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/fsr_rcas.glsl $<TARGET_FILE_DIR:22_gpu_driven>/fsr_rcas.glsl
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/gllib/glsl/ffx_a.h $<TARGET_FILE_DIR:22_gpu_driven>/ffx_a.h
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/gllib/glsl/ffx_fsr1.h $<TARGET_FILE_DIR:22_gpu_driven>/ffx_fsr1.h
)
