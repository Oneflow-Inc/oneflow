include (ExternalProject)

set(HALF_INCLUDE_DIR ${THIRD_PARTY_DIR}/half/include)

set(HALF_URL https://raw.githubusercontent.com/Oneflow-Inc/ThirdPartyLib/master/half-2.1.0.zip)
set(HALF_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/half/src/half)

set(HALF_HEADERS
    "${HALF_BASE_DIR}/include/half.hpp"
)

if(THIRD_PARTY)

ExternalProject_Add(half
    PREFIX half
    URL ${HALF_URL}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

add_custom_target(half_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HALF_INCLUDE_DIR}
    DEPENDS half)

add_custom_target(half_copy_headers_to_destination
    DEPENDS half_create_header_dir)

foreach(header_file ${HALF_HEADERS})
    add_custom_command(TARGET half_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${HALF_INCLUDE_DIR})
endforeach()
endif(THIRD_PARTY)
