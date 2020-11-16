include (ExternalProject)

set(PROTOBUF_INCLUDE_DIR ${THIRD_PARTY_DIR}/protobuf/include)
set(PROTOBUF_LIBRARY_DIR ${THIRD_PARTY_DIR}/protobuf/lib)
set(PROTOBUF_BINARY_DIR ${THIRD_PARTY_DIR}/protobuf/bin)

set(PROTOBUF_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src)
if(WITH_XLA)
  set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
  set(PROTOBUF_MD5 cf02c32870a1f78c860039e0f63a6343)
else()
  set(PROTOBUF_URL https://github.com/Oneflow-Inc/protobuf/archive/1d2c7b6c7.tar.gz)
  set(PROTOBUF_MD5 b859a1c299f1b374b25de2f0df9f9b50)
endif()

use_mirror(VARIABLE PROTOBUF_URL URL ${PROTOBUF_URL})

if(WIN32)
    set(PROTOBUF_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/${CMAKE_BUILD_TYPE})
    set(PROTOBUF_LIBRARY_NAMES libprotobufd.lib)
    set(PROTOC_EXECUTABLE_NAME protoc.exe)
    set(PROTOBUF_ADDITIONAL_CMAKE_OPTIONS -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=ON -A x64)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(PROTOBUF_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf)
    set(PROTOBUF_LIBRARY_NAMES libprotobuf.a)
    set(PROTOC_EXECUTABLE_NAME protoc)
else()
    set(PROTOBUF_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf)
    set(PROTOBUF_LIBRARY_NAMES libprotobuf.a)
    set(PROTOC_EXECUTABLE_NAME protoc)
endif()

foreach(LIBRARY_NAME ${PROTOBUF_LIBRARY_NAMES})
    list(APPEND PROTOBUF_STATIC_LIBRARIES ${PROTOBUF_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND PROTOBUF_BUILD_STATIC_LIBRARIES ${PROTOBUF_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(PROTOBUF_BUILD_PROTOC_EXECUTABLE ${PROTOBUF_BUILD_LIBRARY_DIR}/${PROTOC_EXECUTABLE_NAME})
set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_BINARY_DIR}/${PROTOC_EXECUTABLE_NAME})

if (THIRD_PARTY)

ExternalProject_Add(protobuf
    PREFIX protobuf
    DEPENDS zlib
    URL ${PROTOBUF_URL}
    URL_MD5 ${PROTOBUF_MD5}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
    CONFIGURE_COMMAND ${CMAKE_COMMAND} cmake/
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DZLIB_ROOT=${ZLIB_INSTALL}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        ${PROTOBUF_ADDITIONAL_CMAKE_OPTIONS}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
)

# put protobuf includes in the 'THIRD_PARTY_DIR'
if(WITH_XLA)
  add_copy_headers_target(NAME protobuf SRC ${PROTOBUF_SRC_DIR} DST ${PROTOBUF_INCLUDE_DIR} DEPS protobuf INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/protobuf_xla_headers.txt")
else()
  add_copy_headers_target(NAME protobuf SRC ${PROTOBUF_SRC_DIR} DST ${PROTOBUF_INCLUDE_DIR} DEPS protobuf INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/protobuf_headers.txt")
endif()


# put protobuf librarys in the 'THIRD_PARTY_DIR'
add_custom_target(protobuf_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTOBUF_LIBRARY_DIR}
  DEPENDS protobuf)

add_custom_target(protobuf_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROTOBUF_BUILD_STATIC_LIBRARIES} ${PROTOBUF_LIBRARY_DIR}
  DEPENDS protobuf_create_library_dir)

# pub protoc binary in the 'THIRD_PARTY_DIR'
add_custom_target(protobuf_create_binary_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTOBUF_BINARY_DIR}
  DEPENDS protobuf)

add_custom_target(protobuf_copy_binary_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROTOBUF_BUILD_PROTOC_EXECUTABLE} ${PROTOBUF_BINARY_DIR}
  DEPENDS protobuf_create_binary_dir)

endif(THIRD_PARTY)
