include(ExternalProject)

set(GRPC_INSTALL_DIR ${THIRD_PARTY_DIR}/grpc)
set(GRPC_INSTALL_INCLUDE_DIR include)
set(GRPC_INSTALL_LIBRARY_DIR lib)
set(GRPC_INCLUDE_DIR ${THIRD_PARTY_DIR}/grpc/${GRPC_INSTALL_INCLUDE_DIR})
set(GRPC_LIBRARY_DIR ${THIRD_PARTY_DIR}/grpc/${GRPC_INSTALL_LIBRARY_DIR})

set(GRPC_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/grpc/src/grpc/include)
SET(GRPC_TAR_URL https://github.com/grpc/grpc/archive/v1.27.3.tar.gz)
use_mirror(VARIABLE GRPC_TAR_URL URL ${GRPC_TAR_URL})
set(GRPC_URL_HASH 0c6c3fc8682d4262dd0e5e6fabe1a7e2)
SET(GRPC_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/grpc)

if(WIN32)
    set(GRPC_LIBRARY_NAMES grpc++_unsecure.lib
      grpc_unsecure.lib gpr.lib upb.lib address_sorting.lib cares.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GRPC_LIBRARY_NAMES libgrpc++_unsecure.a
      libgrpc_unsecure.a libgpr.a libupb.a libaddress_sorting.a libcares.a)
else()
    include(GNUInstallDirs)
    set(GRPC_LIBRARY_NAMES libgrpc++_unsecure.a
      libgrpc_unsecure.a libgpr.a libupb.a libaddress_sorting.a libcares.a)
endif()

foreach(LIBRARY_NAME ${GRPC_LIBRARY_NAMES})
    list(APPEND GRPC_STATIC_LIBRARIES ${GRPC_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(PROTOBUF_CONFIG_DIR ${PROTOBUF_LIBRARY_DIR}/cmake/protobuf)
set(ABSL_CONFIG_DIR ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/cmake/absl)

if(THIRD_PARTY)

include(ProcessorCount)
ProcessorCount(PROC_NUM)
ExternalProject_Add(grpc
    PREFIX ${GRPC_SOURCE_DIR}
    DEPENDS protobuf absl cares openssl zlib zlib_copy_headers_to_destination
    URL ${GRPC_TAR_URL}
    URL_HASH MD5=${GRPC_URL_HASH}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j ${PROC_NUM} --target grpc && ${CMAKE_COMMAND} --build . -j ${PROC_NUM} --target grpc_unsecure && ${CMAKE_COMMAND} --build . -j ${PROC_NUM} --target grpc++_unsecure
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
        -DgRPC_INSTALL:BOOL=ON
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:PATH=${ABSL_CONFIG_DIR}
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${PROTOBUF_CONFIG_DIR}
        -DgRPC_CARES_PROVIDER:STRING=module
        -DCARES_ROOT_DIR:PATH=${CARES_SOURCE_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:PATH=${ZLIB_INSTALL}
        -DgRPC_SSL_PROVIDER:STRING=package
        -DOPENSSL_ROOT_DIR:PATH=${OPENSSL_INSTALL}
        -DCMAKE_INSTALL_PREFIX:STRING=${GRPC_INSTALL_DIR}
        -DCMAKE_INSTALL_INCLUDEDIR:STRING=${GRPC_INSTALL_INCLUDE_DIR}
        -DCMAKE_INSTALL_LIBDIR:STRING=${GRPC_INSTALL_LIBRARY_DIR}
)
endif(THIRD_PARTY)
