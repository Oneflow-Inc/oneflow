include(ExternalProject)

set(ONEREC_URL ${CMAKE_CURRENT_BINARY_DIR}/third_party/onerec/src/onerec)
set(ONEREC_INSTALL_PREFIX ${THIRD_PARTY_DIR}/onerec)
set(ONEREC_INCLUDE_DIR ${ONEREC_INSTALL_PREFIX}/include)
set(ONEREC_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onerec/src/onerec)

set(ONEREC_HEADERS
        "${ONEREC_BASE_DIR}/cpp/gen/include/onerec/example_generated.h"
        )

if (THIRD_PARTY)

    ExternalProject_Add(onerec
            PREFIX onerec
            URL ${ONEREC_URL}
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE 1
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            )

    add_custom_target(onerec_create_header_dir
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ONEREC_INCLUDE_DIR}/onerec
            DEPENDS onerec)

    add_custom_target(onerec_copy_headers_to_destination
            DEPENDS onerec_create_header_dir)

    foreach (header_file ${ONEREC_HEADERS})
        add_custom_command(TARGET onerec_copy_headers_to_destination PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${ONEREC_INCLUDE_DIR}/onerec/)
    endforeach ()

endif (THIRD_PARTY)
