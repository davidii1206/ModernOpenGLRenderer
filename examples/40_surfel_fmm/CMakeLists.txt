add_executable(40_surfel_fmm
    main.cpp
    gpu_util.cpp
    surfels.cpp
    grid.cpp
    direct.cpp
    cuts.cpp
    brute.cpp
    fmm.cpp
    screen.cpp
    validate.cpp
)
target_link_libraries(40_surfel_fmm PRIVATE gllib)
target_compile_features(40_surfel_fmm PRIVATE cxx_std_20)

# Shaders get their own always-run target rather than a POST_BUILD command:
# POST_BUILD only fires when the executable relinks, so editing a shader alone
# would never refresh the copy the binary actually loads, and the run would
# silently use the previous version.
#
# The dependency runs exe -> shaders, NOT the other way round. With it reversed,
# `--target 40_surfel_fmm` links new C++ against the PREVIOUSLY copied shaders and
# says nothing about it: a run then validates the old kernel, which during the
# two-layer work showed up as "every number is bit-identical, the change does
# nothing" when in fact the change had never reached the GPU. Copying into
# CMAKE_CURRENT_BINARY_DIR rather than $<TARGET_FILE_DIR:...> is what allows the
# dependency to point this way (the destination no longer needs the exe to exist
# first); the two are the same directory for single-config generators.
add_custom_target(40_surfel_fmm_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
  COMMENT "Copying 40_surfel_fmm shaders"
)
add_dependencies(40_surfel_fmm 40_surfel_fmm_shaders)

# The model and both path-traced references. copy_if_different so a relink does
# not needlessly refresh mtimes.
add_custom_command(TARGET 40_surfel_fmm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb
          ${CMAKE_CURRENT_SOURCE_DIR}/CornellBoxGroundTruthDirectLighting.png
          ${CMAKE_CURRENT_SOURCE_DIR}/CornellBoxOriginalGroundTruth.png
          $<TARGET_FILE_DIR:40_surfel_fmm>
)
