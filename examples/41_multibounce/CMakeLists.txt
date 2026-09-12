add_executable(41_multibounce
    main.cpp
    gpu_util.cpp
    scene.cpp
    solver.cpp
    screen.cpp
    validate.cpp
)
target_link_libraries(41_multibounce PRIVATE gllib)
target_compile_features(41_multibounce PRIVATE cxx_std_20)

# Shaders get their own always-run target rather than a POST_BUILD command:
# POST_BUILD only fires when the executable relinks, so editing a shader alone
# would never refresh the copy the binary actually loads, and the run would
# silently use the previous version. The dependency runs exe -> shaders, not the
# other way round, so `--target 41_multibounce` cannot link new C++ against the
# previously copied shaders -- see example 40's CMakeLists for the incident that
# convention comes from.
add_custom_target(41_multibounce_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
  COMMENT "Copying 41_multibounce shaders"
)
add_dependencies(41_multibounce 41_multibounce_shaders)

# The model and both path-traced references, copied from example 40 so the two
# examples are scored against the same images.
add_custom_command(TARGET 41_multibounce POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb
          ${CMAKE_CURRENT_SOURCE_DIR}/CornellBoxGroundTruthDirectLighting.png
          ${CMAKE_CURRENT_SOURCE_DIR}/CornellBoxOriginalGroundTruth.png
          $<TARGET_FILE_DIR:41_multibounce>
)
