# ------------------------------------------------------------------------------#
#
# TODO: [DFG] Remove this file after 'rocprofsys' rebranding is complete
# rocprofsys-exe target (deprecated 'omnitrace' executable, now 'rocprofsys-instrument')
#
# ------------------------------------------------------------------------------#

add_executable(rocprofsys-exe ${CMAKE_CURRENT_LIST_DIR}/omnitrace.cpp)

target_link_libraries(rocprofsys-exe PRIVATE rocprofsys::rocprofsys-threading)

set_target_properties(
    rocprofsys-exe
    PROPERTIES OUTPUT_NAME omnitrace
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH ON)

omnitrace_strip_target(rocprofsys-exe)

install(
    TARGETS rocprofsys-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
