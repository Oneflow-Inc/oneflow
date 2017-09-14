include (ExternalProject)

set(GRPC_INCLUDE_DIR ${THIRD_PARTY_DIR}/grpc/include)
set(GRPC_LIBRARY_DIR ${THIRD_PARTY_DIR}/grpc/lib)
set(GRPC_BINARY_DIR  ${THIRD_PARTY_DIR}/grpc/bin)

set(GRPC_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc/include)
set(GRPC_URL https://github.com/grpc/grpc.git)
set(GRPC_BUILD ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc)
set(GRPC_TAG 781fd6f6ea03645a520cd5c675da67ab61f87e4b)

if(WIN32)
  set(GRPC_BUILD_LIBRARY_DIR ${GRPC_BUILD}/${CMAKE_BUILD_TYPE})
    set(GRPC_LIBRARY_NAMES gpr.lib
      grpc_unsecure.lib
      grpc++_unsecure.lib)
    set(GRPC_CPP_PLUGIN_NAME grpc_cpp_plugin.exe)
else()
  set(GRPC_BUILD_LIBRARY_DIR ${GRPC_BUILD})
    set(GRPC_LIBRARY_NAMES libgpr.a
      libgrpc_unsecure.a
      libgrpc++_unsecure.a)
    set(GRPC_CPP_PLUGIN_NAME grpc_cpp_plugin)
endif()

set(GRPC_CPP_PLUGIN_PATH ${GRPC_BINARY_DIR}/${GRPC_CPP_PLUGIN_NAME})

foreach(LIBRARY_NAME ${GRPC_LIBRARY_NAMES})
    list(APPEND GRPC_STATIC_LIBRARIES ${GRPC_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GRPC_BUILD_STATIC_LIBRARIES ${GRPC_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(BUILD_THIRD_PARTY)

ExternalProject_Add(grpc
    PREFIX grpc
    DEPENDS protobuf zlib
    GIT_REPOSITORY ${GRPC_URL}
    GIT_TAG ${GRPC_TAG}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target grpc++_unsecure 
    COMMAND ${CMAKE_COMMAND} --build . --config Release --target grpc_cpp_plugin
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/grpc/CMakeLists.txt ${GRPC_BUILD}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DPROTOBUF_INCLUDE_DIRS:STRING=${PROTOBUF_SRC_DIR}
        -DPROTOBUF_LIBRARIES:STRING=${protobuf_STATIC_LIBRARIES}
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
        -DgRPC_SSL_PROVIDER:STRING=NONE
)

ExternalProject_Add_Step(grpc copy_rand
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/cmake/patches/grpc/rand.h ${GRPC_BUILD}/include/openssl/rand.h
    DEPENDEES patch
    DEPENDERS build
)

add_custom_target(grpc_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRPC_INCLUDE_DIR}
  DEPENDS grpc)

add_custom_target(grpc_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${GRPC_INCLUDE_DIRS} ${GRPC_INCLUDE_DIR}
  DEPENDS grpc_create_header_dir)

add_custom_target(grpc_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRPC_LIBRARY_DIR}
  DEPENDS grpc)

add_custom_target(grpc_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GRPC_BUILD_STATIC_LIBRARIES} ${GRPC_LIBRARY_DIR}
  DEPENDS grpc_create_library_dir)

add_custom_target(grpc_create_binary_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GRPC_BINARY_DIR}
  DEPENDS grpc)

add_custom_target(grpc_copy_binary_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GRPC_BUILD_LIBRARY_DIR}/${GRPC_CPP_PLUGIN_NAME} ${GRPC_CPP_PLUGIN_PATH}
  DEPENDS grpc_create_binary_dir)

endif(BUILD_THIRD_PARTY)
