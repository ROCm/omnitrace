#
set(binary_sources
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.cpp
    ${CMAKE_CURRENT_LIST_DIR}/address_range.cpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bfd_line_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_line_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.cpp)

set(binary_headers
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.hpp
    ${CMAKE_CURRENT_LIST_DIR}/address_range.hpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.hpp
    ${CMAKE_CURRENT_LIST_DIR}/basic_line_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/bfd_line_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_line_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/line_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.hpp)

target_sources(omnitrace-object-library PRIVATE ${binary_sources} ${binary_headers})
