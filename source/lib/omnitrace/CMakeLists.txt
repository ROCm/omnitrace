# ------------------------------------------------------------------------------#
#
# rocprofsys object library
#
# ------------------------------------------------------------------------------#

add_library(rocprofsys-object-library OBJECT)
add_library(rocprofsys::rocprofsys-object-library ALIAS rocprofsys-object-library)

target_sources(
    rocprofsys-object-library
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/library.cpp ${CMAKE_CURRENT_LIST_DIR}/regions.cpp
            ${CMAKE_CURRENT_LIST_DIR}/progress.cpp ${CMAKE_CURRENT_LIST_DIR}/api.cpp
            ${CMAKE_CURRENT_LIST_DIR}/timeout.cpp ${CMAKE_CURRENT_LIST_DIR}/api.hpp)

add_subdirectory(library)

target_link_libraries(rocprofsys-object-library
                      PRIVATE rocprofsys::rocprofsys-interface-library)

# ------------------------------------------------------------------------------#
#
# rocprofsys static library
#
# ------------------------------------------------------------------------------#

add_library(rocprofsys-static-library STATIC $<TARGET_OBJECTS:rocprofsys-object-library>)
add_library(rocprofsys::librocprofsys-static ALIAS rocprofsys-static-library)

target_link_libraries(
    rocprofsys-static-library
    PRIVATE rocprofsys::rocprofsys-interface-library rocprofsys::rocprofsys-core
            rocprofsys::rocprofsys-binary)

set_target_properties(rocprofsys-static-library PROPERTIES OUTPUT_NAME rocprof-sys)

# ------------------------------------------------------------------------------#
#
# rocprofsys shared library
#
# ------------------------------------------------------------------------------#

add_library(rocprofsys-shared-library SHARED $<TARGET_OBJECTS:rocprofsys-object-library>)
add_library(rocprofsys::librocprofsys-shared ALIAS rocprofsys-shared-library)
add_library(rocprofsys::rocprofsys-library ALIAS rocprofsys-shared-library)

target_link_libraries(
    rocprofsys-shared-library
    PRIVATE rocprofsys::rocprofsys-interface-library rocprofsys::rocprofsys-core
            rocprofsys::rocprofsys-binary)

set_target_properties(
    rocprofsys-shared-library
    PROPERTIES OUTPUT_NAME rocprof-sys
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               INSTALL_RPATH "${OMNITRACE_LIB_INSTALL_RPATH}")

omnitrace_strip_target(rocprofsys-shared-library)

install(
    TARGETS rocprofsys-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
