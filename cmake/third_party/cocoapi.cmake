include (ExternalProject)

set(COCOAPI_INCLUDE_DIR ${THIRD_PARTY_DIR}/cocoapi/include)
set(COCOAPI_LIBRARY_DIR ${THIRD_PARTY_DIR}/cocoapi/lib)

set(COCOAPI_URL https://github.com/cocodataset/cocoapi.git)
set(COCOAPI_TAG ed842bffd41f6ff38707c4f0968d2cfd91088688)
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
    GIT_REPOSITORY ${COCOAPI_URL}
    GIT_TAG ${COCOAPI_TAG}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND gcc -fPIC -O3 -c common/maskApi.c -o maskApi.o && ar rcs ${COCOAPI_LIBRARY_NAME} maskApi.o
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
