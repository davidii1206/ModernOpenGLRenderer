add_executable(39_mosaic
    main.cpp
    gpu_util.cpp
    screen.cpp
    surfel_bake.cpp
    clipmap.cpp
    cache_gi.cpp
    direct.cpp
)
target_link_libraries(39_mosaic PRIVATE gllib)
target_compile_features(39_mosaic PRIVATE cxx_std_20)

# Shaders get their own always-run target rather than a POST_BUILD command:
# POST_BUILD only fires when the executable relinks, so editing a shader alone
# would never refresh the copy the binary actually loads, and the run would
# silently use the previous version.
add_custom_target(39_mosaic_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:39_mosaic>/shaders
  COMMENT "Copying 39_mosaic shaders"
)
add_dependencies(39_mosaic_shaders 39_mosaic)

# copy_if_different, not copy: the surfel cache is keyed on the model file's
# mtime, and an unconditional copy refreshes that on every relink, invalidating
# every cached surfel set and forcing a full rebake after any code change.
add_custom_command(TARGET 39_mosaic POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_SOURCE_DIR}/data/CornellBoxOriginal.glb
          ${CMAKE_SOURCE_DIR}/data/sponza.glb
          ${CMAKE_SOURCE_DIR}/data/Stanford_Bunny.glb
          $<TARGET_FILE_DIR:39_mosaic>
)
