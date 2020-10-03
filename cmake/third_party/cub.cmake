include (ExternalProject)

set(CUB_INCLUDE_DIR ${THIRD_PARTY_DIR}/cub/include)
set(CUB_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/cub/src/cub/cub)

if(THIRD_PARTY)

ExternalProject_Add(cub
    PREFIX cub
    URL https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz
    URL_MD5 9203ea2499b56782601fddf8a12e9b08
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

add_copy_headers_target(NAME cub SRC ${CUB_BUILD_INCLUDE} DST ${CUB_INCLUDE_DIR}/cub DEPS cub INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/cub_headers.txt")

endif(THIRD_PARTY)
