
include (ExternalProject)

set(PYBIND11_URL https://github.com/Oneflow-Inc/pybind11/archive/v2.5.0.tar.gz)
set(PYBIND11_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pybind11/src/pybind11)
set(PYBIND11_INSTALL_DIR ${THIRD_PARTY_DIR}/pybind11)
SET(PYBIND11_INCLUDE_DIR ${PYBIND11_INSTALL_DIR}/include/pybind11 CACHE PATH "" FORCE)

SET(PYBIND11_HEADERS
    "${PYBIND11_BASE_DIR}/include/pybind11/embed.h"
    "${PYBIND11_BASE_DIR}/include/pybind11/numpy.h"
    "${PYBIND11_BASE_DIR}/include/pybind11/stl.h"
)

if(THIRD_PARTY)
    ExternalProject_Add(pybind11
        PREFIX pybind11
        URL ${PYBIND11_URL}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )
    add_custom_target(pybind11_create_header_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PYBIND11_INCLUDE_DIR}
        DEPENDS pybind11
    )
    add_custom_target(pybind11_copy_headers_to_destination
        DEPENDS pybind11_create_header_dir
    )
    foreach(header_file ${PYBIND11_HEADERS})
        add_custom_command(TARGET pybind11_copy_headers_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${PYBIND11_INCLUDE_DIR}
        )
    endforeach()
endif(THIRD_PARTY)