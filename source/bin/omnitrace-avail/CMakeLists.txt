# ------------------------------------------------------------------------------#
#
# rocprofsys-avail target
#
# ------------------------------------------------------------------------------#

add_executable(rocprofsys-avail)

target_sources(
    rocprofsys-avail
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/avail.cpp
            ${CMAKE_CURRENT_LIST_DIR}/avail.hpp
            ${CMAKE_CURRENT_LIST_DIR}/common.cpp
            ${CMAKE_CURRENT_LIST_DIR}/common.hpp
            ${CMAKE_CURRENT_LIST_DIR}/component_categories.hpp
            ${CMAKE_CURRENT_LIST_DIR}/defines.hpp
            ${CMAKE_CURRENT_LIST_DIR}/enumerated_list.hpp
            ${CMAKE_CURRENT_LIST_DIR}/generate_config.cpp
            ${CMAKE_CURRENT_LIST_DIR}/generate_config.hpp
            ${CMAKE_CURRENT_LIST_DIR}/get_availability.hpp
            ${CMAKE_CURRENT_LIST_DIR}/get_categories.hpp
            ${CMAKE_CURRENT_LIST_DIR}/info_type.cpp
            ${CMAKE_CURRENT_LIST_DIR}/info_type.hpp)

target_include_directories(rocprofsys-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(rocprofsys-avail PRIVATE OMNITRACE_EXTERN_COMPONENTS=0)
target_link_libraries(
    rocprofsys-avail
    PRIVATE rocprofsys::rocprofsys-compile-definitions
            rocprofsys::rocprofsys-interface-library rocprofsys::librocprofsys-static)
set_target_properties(
    rocprofsys-avail
    PROPERTIES BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               OUTPUT_NAME rocprof-sys-avail)

omnitrace_strip_target(rocprofsys-avail)

install(
    TARGETS rocprofsys-avail
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
