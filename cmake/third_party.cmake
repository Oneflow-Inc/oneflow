set(THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party"
  CACHE PATH "Location where third party headers and libs will be put.")
mark_as_advanced(THIRD_PARTY_DIR)

if (NOT WIN32)
  find_package(Threads)
endif()

if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC -D__VERSION__=\"MSVC\")
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

include(zlib)
include(protobuf)
include(googletest)
include(glog)
include(gflags)
include(grpc)

if(WIN32)
  include(netdirect)
endif()

set(oneflow_third_party_libs
    ${CMAKE_THREAD_LIBS_INIT}
    ${ZLIB_STATIC_LIBRARIES}
    ${GLOG_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${PROTOBUF_STATIC_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
)

if(WIN32)
  # static gflags lib requires "PathMatchSpecA" defined in "ShLwApi.Lib"
  list(APPEND oneflow_third_party_libs "ShLwApi.Lib")
endif()

set(oneflow_third_party_dependencies
  zlib_copy_headers_to_destination
  zlib_copy_libs_to_destination
  gflags_copy_headers_to_destination
  gflags_copy_libs_to_destination
  glog_copy_headers_to_destination
  glog_copy_libs_to_destination
  googletest_copy_headers_to_destination
  googletest_copy_libs_to_destination
  protobuf_copy_headers_to_destination
  protobuf_copy_libs_to_destination
  protobuf_copy_binary_to_destination
  grpc_copy_headers_to_destination
  grpc_copy_libs_to_destination
)

include_directories(
    ${ZLIB_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${GRPC_INCLUDE_DIR}
)

if(WIN32)
  list(APPEND oneflow_third_party_dependencies netdirect)
  include_directories(${NETDIRECT_INCLUDE_DIR})
endif()