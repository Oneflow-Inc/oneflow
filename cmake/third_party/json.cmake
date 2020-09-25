include(ExternalProject)

SET(JSON_URL https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip)
use_mirror(VARIABLE JSON_URL URL ${JSON_URL})
SET(JSON_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/json/src/json)
SET(JSON_INSTALL_DIR ${THIRD_PARTY_DIR}/json)
SET(JSON_INCLUDE_DIR ${JSON_INSTALL_DIR}/include CACHE PATH "" FORCE)
SET(JSON_URL_HASH fb96f95cdf609143e998db401ca4f324)
SET(JSON_HEADERS
    "${JSON_BASE_DIR}/single_include/nlohmann/json.hpp"
)

if(THIRD_PARTY)
    ExternalProject_Add(json
        PREFIX json
        URL ${JSON_URL}
        URL_HASH MD5=${JSON_URL_HASH}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )
    add_custom_target(json_create_header_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${JSON_INCLUDE_DIR}
        DEPENDS json
    )
    add_custom_target(json_copy_headers_to_destination
        DEPENDS json_create_header_dir
    )
    foreach(header_file ${JSON_HEADERS})
        add_custom_command(TARGET json_copy_headers_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${JSON_INCLUDE_DIR}
        )
    endforeach()
endif(THIRD_PARTY)
