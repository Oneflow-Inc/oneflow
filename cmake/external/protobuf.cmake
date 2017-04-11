include (ExternalProject)

set(PROTOBUF_INCLUDE_DIR ${THIRD_PARTY_DIR}/protobuf/include)
set(PROTOBUF_LIBRARY_DIR ${THIRD_PARTY_DIR}/protobuf/lib)
set(PROTOBUF_BINARY_DIR ${THIRD_PARTY_DIR}/protobuf/bin)

set(PROTOBUF_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/src)
set(PROTOBUF_URL https://github.com/mrry/protobuf.git)  # Includes MSVC fix.
set(PROTOBUF_TAG 1d2c7b6c7376f396c8c7dd9b6afd2d4f83f3cb05)

if(WIN32)
    set(PROTOBUF_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/${CMAKE_BUILD_TYPE})
    set(PROTOBUF_LIBRARY_NAMES libprotobufd.lib)
    set(PROTOC_EXECUTABLE_NAME protoc.exe)
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
set(PROTOBUF_PROTOC ${PROTOBUF_BINARY_DIR}/${PROTOC_EXECUTABLE_NAME})

ExternalProject_Add(protobuf
    PREFIX protobuf
    DEPENDS zlib
    GIT_REPOSITORY ${PROTOBUF_URL}
    GIT_TAG ${PROTOBUF_TAG}
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
    CONFIGURE_COMMAND ${CMAKE_COMMAND} cmake/
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DZLIB_ROOT=${ZLIB_INSTALL}
        ${PROTOBUF_ADDITIONAL_CMAKE_OPTIONS}
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
)

# put protobuf includes in the 'THIRD_PARTY_DIR'
add_custom_target(protobuf_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROTOBUF_INCLUDE_DIR}
  DEPENDS protobuf)

add_custom_target(protobuf_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROTOBUF_SRC_DIR} ${PROTOBUF_INCLUDE_DIR}
  DEPENDS protobuf_create_header_dir)

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