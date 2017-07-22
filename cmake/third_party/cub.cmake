include (ExternalProject)

set(CUB_INCLUDE_DIR ${THIRD_PARTY_DIR}/cub/include)
set(CUB_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/cub/src/cub/cub)

set(CUB_URL https://github.com/Oneflow-Inc/cub.git)
set(CUB_TAG 01347a797c620618d09e7d2d90bce4be4c42513e)

if(BUILD_THIRD_PARTY)

ExternalProject_Add(cub
    PREFIX cub
    GIT_REPOSITORY ${CUB_URL}
    GIT_TAG ${CUB_TAG}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

add_custom_target(cub_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CUB_INCLUDE_DIR}/cub
    DEPENDS cub)
add_custom_target(cub_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CUB_BUILD_INCLUDE} ${CUB_INCLUDE_DIR}/cub
    DEPENDS cub_create_header_dir)

endif(BUILD_THIRD_PARTY)