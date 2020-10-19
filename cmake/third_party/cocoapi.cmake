include (ExternalProject)

set(COCOAPI_INCLUDE_DIR ${THIRD_PARTY_DIR}/cocoapi/include)
set(COCOAPI_LIBRARY_DIR ${THIRD_PARTY_DIR}/cocoapi/lib)

set(COCOAPI_URL https://github.com/Oneflow-Inc/cocoapi/archive/ed842bf.tar.gz)
use_mirror(VARIABLE COCOAPI_URL URL ${COCOAPI_URL})
set(COCOAPI_URL_HASH e7e0504231e5614ffaa34f081773f7f1)
set(COCOAPI_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cocoapi/src/cocoapi)
set(COCOAPI_LIBRARY_NAME libcocoapi_static.a)

list(APPEND COCOAPI_STATIC_LIBRARIES ${COCOAPI_LIBRARY_DIR}/${COCOAPI_LIBRARY_NAME})
list(APPEND COCOAPI_BUILD_STATIC_LIBRARIES ${COCOAPI_BASE_DIR}/${COCOAPI_LIBRARY_NAME})

set(COCOAPI_HEADERS
    "${COCOAPI_BASE_DIR}/common/maskApi.h"
)

if(THIRD_PARTY)

ExternalProject_Add(cocoapi
    PREFIX cocoapi
    URL ${COCOAPI_URL}
    URL_HASH MD5=${COCOAPI_URL_HASH}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_C_COMPILER} -fPIC -O3 -c common/maskApi.c -o maskApi.o &&
        ${CMAKE_AR} rcs ${COCOAPI_LIBRARY_NAME} maskApi.o
    INSTALL_COMMAND ""
)

add_custom_target(cocoapi_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${COCOAPI_INCLUDE_DIR}
    DEPENDS cocoapi)

add_custom_target(cocoapi_copy_headers_to_destination
    DEPENDS cocoapi_create_header_dir)

foreach(header_file ${COCOAPI_HEADERS})
    add_custom_command(TARGET cocoapi_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${COCOAPI_INCLUDE_DIR})
endforeach()

add_custom_target(cocoapi_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${COCOAPI_LIBRARY_DIR}
    DEPENDS cocoapi)

add_custom_target(cocoapi_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${COCOAPI_BUILD_STATIC_LIBRARIES} ${COCOAPI_LIBRARY_DIR}
    DEPENDS cocoapi_create_library_dir)
endif(THIRD_PARTY)
