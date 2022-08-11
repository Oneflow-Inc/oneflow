find_package(Threads REQUIRED)
set(ONETBB_INSTALL_DIR ${THIRD_PARTY_DIR}/tbb CACHE PATH " ")

include(FetchContent)
FetchContent_Declare(
  tbb
  URL ${ONETBB_URL}
  URL_HASH MD5=${ONETBB_MD5}
)
FetchContent_GetProperties(tbb)

set(TBB_EXAMPLES OFF CACHE BOOL "")
set(TBB_TEST OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)

FetchContent_MakeAvailable(tbb)

install(TARGETS tbb tbbmalloc tbbmalloc_proxy COMPONENT OneFlowTBB)
install(DIRECTORY ${tbb_SOURCE_DIR}/include DESTINATION ${ONETBB_INSTALL_DIR} COMPONENT OneFlowTBB)

add_custom_target(install-tbb
  DEPENDS tbb tbbmalloc tbbmalloc_proxy
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_PREFIX=${ONETBB_INSTALL_DIR}
      -DCMAKE_INSTALL_COMPONENT=OneFlowTBB
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)
