include (ExternalProject)

set(zlib_INCLUDE_DIR ${THIRD_PARTY_DIR}/zlib/include)
set(zlib_LIBRARY_DIR ${THIRD_PARTY_DIR}/zlib/lib)

set(ZLIB_URL https://github.com/madler/zlib)
set(ZLIB_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/zlib/install)
set(ZLIB_TAG 50893291621658f355bc5b4d450a8d06a563053d)

if(WIN32)
  set(zlib_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/zlibstaticd.lib)
else()
  set(zlib_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib/libz.a)
endif()

set(ZLIB_HEADERS
    "${ZLIB_INSTALL}/include/zconf.h"
    "${ZLIB_INSTALL}/include/zlib.h"
)

ExternalProject_Add(zlib
    PREFIX zlib
    GIT_REPOSITORY ${ZLIB_URL}
    GIT_TAG ${ZLIB_TAG}
    INSTALL_DIR ${ZLIB_INSTALL}
    BUILD_IN_SOURCE 1
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${ZLIB_INSTALL}
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

# put zlib includes in the directory where they are expected
add_custom_target(zlib_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${zlib_INCLUDE_DIR}
    DEPENDS zlib)

add_custom_target(zlib_copy_headers_to_destination
    DEPENDS zlib_create_header_dir)

foreach(header_file ${ZLIB_HEADERS})
    add_custom_command(TARGET zlib_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${zlib_INCLUDE_DIR})
endforeach()

# pub zlib libs in the directory where they are expected
add_custom_target(zlib_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${zlib_LIBRARY_DIR}
    DEPENDS zlib)

add_custom_target(zlib_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${zlib_STATIC_LIBRARIES} ${zlib_LIBRARY_DIR}
    DEPENDS zlib_create_library_dir)
