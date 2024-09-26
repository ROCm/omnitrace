# ------------------------------------------------------------------------------#
#
# rocprofsys-run target
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofsys-run
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-run.cpp
    ${CMAKE_CURRENT_LIST_DIR}/omnitrace-run.hpp ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(rocprofsys-run PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofsys-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofsys-run
    PRIVATE rocprofsys::rocprofsys-compile-definitions rocprofsys::rocprofsys-headers
            rocprofsys::rocprofsys-common-library rocprofsys::rocprofsys-core
            rocprofsys::rocprofsys-sanitizer)
set_target_properties(
    rocprofsys-run
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               OUTPUT_NAME "rocprof-sys-run")

omnitrace_strip_target(rocprofsys-run)

install(
    TARGETS rocprofsys-run
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
