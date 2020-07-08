include (ExternalProject)

set(CUB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub/src/cub)

set(CUB_URL ${THIRD_PARTY_SUBMODULE_DIR}/cub/src/cub)

if(THIRD_PARTY)

ExternalProject_Add(cub
    PREFIX cub
    URL ${CUB_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

endif(THIRD_PARTY)
