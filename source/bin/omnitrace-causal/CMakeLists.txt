# ------------------------------------------------------------------------------#
#
# rocprofsys-causal target
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofsys-causal
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-causal.hpp ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(rocprofsys-causal PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofsys-causal PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofsys-causal
    PRIVATE rocprofsys::rocprofsys-compile-definitions rocprofsys::rocprofsys-headers
            rocprofsys::rocprofsys-common-library rocprofsys::rocprofsys-core)
set_target_properties(
    rocprofsys-causal
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               OUTPUT_NAME "rocprof-sys-causal")

omnitrace_strip_target(rocprofsys-causal)

install(
    TARGETS rocprofsys-causal
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
