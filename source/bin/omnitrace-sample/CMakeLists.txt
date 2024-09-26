# ------------------------------------------------------------------------------#
#
# rocprofsys-sample target
#
# ------------------------------------------------------------------------------#

add_executable(rocprofsys-sample ${CMAKE_CURRENT_LIST_DIR}/omnitrace-sample.cpp
                                ${CMAKE_CURRENT_LIST_DIR}/impl.cpp)

target_compile_definitions(rocprofsys-sample PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofsys-sample PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofsys-sample
    PRIVATE rocprofsys::rocprofsys-compile-definitions rocprofsys::rocprofsys-headers
            rocprofsys::rocprofsys-common-library)
set_target_properties(
    rocprofsys-sample PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                                INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
                                OUTPUT_NAME "rocprof-sys-sample")

omnitrace_strip_target(rocprofsys-sample)

install(
    TARGETS rocprofsys-sample
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
