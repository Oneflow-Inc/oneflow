
include (ExternalProject)

set(PYBIND11_URL https://github.com/Oneflow-Inc/pybind11/archive/v2.5.0.tar.gz)
set(PYBIND11_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pybind11/src/pybind11)
set(PYBIND11_INSTALL_DIR ${THIRD_PARTY_DIR}/pybind11)
SET(PYBIND11_INCLUDE_DIR ${PYBIND11_INSTALL_DIR}/include/ CACHE PATH "" FORCE)

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

add_copy_headers_target(NAME pybind11 SRC ${PYBIND11_BASE_DIR} DST ${PYBIND11_INCLUDE_DIR} DEPS pybind11 INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/pybind11_headers.txt")

endif(THIRD_PARTY)