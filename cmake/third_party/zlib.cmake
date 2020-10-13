include (ExternalProject)

set(ZLIB_INCLUDE_DIR ${THIRD_PARTY_DIR}/zlib/include)
set(ZLIB_LIBRARY_DIR ${THIRD_PARTY_DIR}/zlib/lib)

set(ZLIB_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/zlib/install)
set(ZLIB_URL https://github.com/madler/zlib/archive/v1.2.8.tar.gz)
use_mirror(VARIABLE ZLIB_URL URL ${ZLIB_URL})

if(WIN32)
    set(ZLIB_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib)
    set(ZLIB_LIBRARY_NAMES zlibstaticd.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(ZLIB_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib)
    set(ZLIB_LIBRARY_NAMES libz.a)
else()
    set(ZLIB_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib/install/lib)
    set(ZLIB_LIBRARY_NAMES libz.a)
endif()

foreach(LIBRARY_NAME ${ZLIB_LIBRARY_NAMES})
    list(APPEND ZLIB_STATIC_LIBRARIES ${ZLIB_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND ZLIB_BUILD_STATIC_LIBRARIES ${ZLIB_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(ZLIB_HEADERS
    "${ZLIB_INSTALL}/include/zconf.h"
    "${ZLIB_INSTALL}/include/zlib.h"
)

if(THIRD_PARTY)

ExternalProject_Add(zlib
    PREFIX zlib
    URL ${ZLIB_URL}
    URL_MD5 1eabf2698dc49f925ce0ffb81397098f
    UPDATE_COMMAND ""
    INSTALL_DIR ${ZLIB_INSTALL}
    BUILD_IN_SOURCE 1
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${ZLIB_INSTALL}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

# put zlib includes in the directory where they are expected
add_custom_target(zlib_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ZLIB_INCLUDE_DIR}
    DEPENDS zlib)

add_custom_target(zlib_copy_headers_to_destination
    DEPENDS zlib_create_header_dir)

foreach(header_file ${ZLIB_HEADERS})
    add_custom_command(TARGET zlib_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${ZLIB_INCLUDE_DIR})
endforeach()

# pub zlib libs in the directory where they are expected
add_custom_target(zlib_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ZLIB_LIBRARY_DIR}
    DEPENDS zlib)

add_custom_target(zlib_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ZLIB_BUILD_STATIC_LIBRARIES} ${ZLIB_LIBRARY_DIR}
    DEPENDS zlib_create_library_dir)

endif(THIRD_PARTY)
