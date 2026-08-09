add_executable(18_indirect main.cpp)
target_link_libraries(18_indirect PRIVATE gllib)
target_compile_features(18_indirect PRIVATE cxx_std_20)
