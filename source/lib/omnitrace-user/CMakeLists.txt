# ------------------------------------------------------------------------------#
#
# rocprofsys user library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(rocprofsys-user-library SHARED)
add_library(rocprofsys::rocprofsys-user-library ALIAS rocprofsys-user-library)

set(_user_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/user.h
    ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/causal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/categories.h)

set(_user_sources ${CMAKE_CURRENT_SOURCE_DIR}/user.cpp)

target_sources(rocprofsys-user-library PRIVATE ${_user_sources} ${_user_headers})
target_include_directories(
    rocprofsys-user-library PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    rocprofsys-user-library
    PRIVATE $<BUILD_INTERFACE:rocprofsys::rocprofsys-compile-definitions>)

set_target_properties(
    rocprofsys-user-library
    PROPERTIES OUTPUT_NAME rocprof-sys-user
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

omnitrace_strip_target(rocprofsys-user-library)

foreach(_HEADER ${_user_headers})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/omnitrace/"
                   "${CMAKE_CURRENT_SOURCE_DIR}/${PACKAGE_NAME}/" _DEST ${_HEADER})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/"
                   "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/" _DEST "${_DEST}")
    message(STATUS "Copying ${_HEADER} to ${_DEST}")
    configure_file(${_HEADER} ${_DEST} COPYONLY)
endforeach()

install(FILES ${_user_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE_NAME})

install(
    TARGETS rocprofsys-user-library
    EXPORT rocprofsys-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
