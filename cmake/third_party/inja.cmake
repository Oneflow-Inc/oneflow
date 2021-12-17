include(ExternalProject)

SET(INJA_URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.zip)
# use_mirror(VARIABLE INJA_URL URL ${INJA_URL})
SET(INJA_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/inja/src/inja)
SET(INJA_INSTALL_DIR ${THIRD_PARTY_DIR}/inja)
SET(INJA_INCLUDE_DIR ${INJA_INSTALL_DIR}/include CACHE PATH "" FORCE)
SET(INJA_URL_HASH 611e6b7206d0fb89728a3879f78b4775)
SET(INJA_HEADERS
    "${INJA_BASE_DIR}/single_include/inja/inja.hpp"
)

if(THIRD_PARTY)
    ExternalProject_Add(inja
        PREFIX inja
        URL ${INJA_URL}
        URL_HASH MD5=${INJA_URL_HASH}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )
    add_custom_target(inja_create_header_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${INJA_INCLUDE_DIR}
        DEPENDS inja json
    )
    add_custom_target(inja_copy_headers_to_destination
        DEPENDS inja_create_header_dir json_copy_headers_to_destination
    )
    foreach(header_file ${INJA_HEADERS})
        add_custom_command(TARGET inja_copy_headers_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${INJA_INCLUDE_DIR}
        )
    endforeach()
endif(THIRD_PARTY)
