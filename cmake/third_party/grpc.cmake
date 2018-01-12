include (ExternalProject)

set(GRPC_INCLUDE_DIR ${THIRD_PARTY_DIR}/grpc/include)
set(GRPC_LIBRARY_DIR ${THIRD_PARTY_DIR}/grpc/lib)

set(GRPC_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc/include)
set(GRPC_URL https://github.com/yuanms2/grpc.git)
set(GRPC_TAG e0db46e140405f0f94f03c9a55b302e39a514c48)

if(WIN32)
    set(GRPC_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc/${CMAKE_BUILD_TYPE})
    set(GRPC_LIBRARY_NAMES grpc++_unsecure.lib 
      grpc_unsecure.lib gpr.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GRPC_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc/${CMAKE_BUILD_TYPE})
    set(GRPC_LIBRARY_NAMES libgrpc++_unsecure.a 
      libgrpc_unsecure.a libgpr.a)
else()
    set(GRPC_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc)
    set(GRPC_LIBRARY_NAMES libgrpc++_unsecure.a 
      libgrpc_unsecure.a libgpr.a)
endif()

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
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DPROTOBUF_INCLUDE_DIRS:STRING=${PROTOBUF_SRC_DIR}
        -DPROTOBUF_LIBRARIES:STRING=${protobuf_STATIC_LIBRARIES}
        -DZLIB_INCLUDE_DIRS:STRING=${ZLIB_INCLUDE_DIR}
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

endif(BUILD_THIRD_PARTY)
