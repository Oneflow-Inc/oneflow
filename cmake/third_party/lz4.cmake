include (ExternalProject)

set(LZ4_INCLUDE_DIR ${THIRD_PARTY_DIR}/lz4/include)
set(LZ4_LIBRARY_DIR ${THIRD_PARTY_DIR}/lz4/lib)

set(LZ4_URL https://github.com/lz4/lz4/archive/v1.9.2.tar.gz)
set(LZ4_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/lz4/src/lz4/lib)

set(LZ4_BUILD_LIBRARY_DIR ${LZ4_BUILD_DIR})
set(LZ4_LIBRARY_NAMES liblz4.a)


foreach(LIBRARY_NAME ${LZ4_LIBRARY_NAMES})
    list(APPEND LZ4_STATIC_LIBRARIES ${LZ4_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND LZ4_BUILD_STATIC_LIBRARIES ${LZ4_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(LZ4_HEADERS
    "${LZ4_BUILD_DIR}/lz4frame.h"
    "${LZ4_BUILD_DIR}/lz4frame_static.h"
    "${LZ4_BUILD_DIR}/lz4.h"
    "${LZ4_BUILD_DIR}/lz4hc.h"
    "${LZ4_BUILD_DIR}/xxhash.h"
)

set(LZ4_CFLAGS "-O3 -fPIC")

if(THIRD_PARTY)

ExternalProject_Add(lz4
    PREFIX lz4
    URL ${LZ4_URL}
    URL_MD5 3898c56c82fb3d9455aefd48db48eaad
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j lib CFLAGS=${LZ4_CFLAGS}
    INSTALL_COMMAND ""
)

add_custom_target(lz4_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LZ4_INCLUDE_DIR}
    DEPENDS lz4)

add_custom_target(lz4_copy_headers_to_destination
    DEPENDS lz4_create_header_dir)

foreach(header_file ${LZ4_HEADERS})
    add_custom_command(TARGET lz4_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${LZ4_INCLUDE_DIR})
endforeach()

add_custom_target(lz4_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LZ4_LIBRARY_DIR}
    DEPENDS lz4)

message(STATUS  ${LZ4_BUILD_STATIC_LIBRARIES})
add_custom_target(lz4_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LZ4_BUILD_STATIC_LIBRARIES} ${LZ4_LIBRARY_DIR}
    DEPENDS lz4_create_library_dir)

endif(THIRD_PARTY)
