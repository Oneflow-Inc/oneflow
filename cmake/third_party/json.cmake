include (ExternalProject)

set(JSON_INCLUDE_DIR ${THIRD_PARTY_DIR}/json/include)

set(JSON_URL https://github.com/nlohmann/json.git)
set(JSON_TAG 1126c9ca74fdea22d2ce3a065ac0fcb5792cbdaf)
set(JSON_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/json/src/json)

set(JSON_HEADERS
    "${JSON_BASE_DIR}/single_include/nlohmann/json.hpp"
)

if(THIRD_PARTY)

ExternalProject_Add(json
    PREFIX json
    GIT_REPOSITORY ${JSON_URL}
    GIT_TAG ${JSON_TAG}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

add_custom_target(json_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JSON_INCLUDE_DIR}
    DEPENDS json)

add_custom_target(json_copy_headers_to_destination
    DEPENDS json_create_header_dir)

foreach(header_file ${JSON_HEADERS})
    add_custom_command(TARGET json_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${JSON_INCLUDE_DIR})
endforeach()
endif(THIRD_PARTY)
