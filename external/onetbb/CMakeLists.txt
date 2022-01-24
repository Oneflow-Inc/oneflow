find_package(Threads REQUIRED)
set(ONETBB_INSTALL_DIR ${THIRD_PARTY_DIR}/tbb CACHE PATH " ")

include(FetchContent)
FetchContent_Declare(
  tbb
  URL ${ONETBB_URL}
  # URL_HASH MD5=${ONETBB_MD5}
  URL_MD5 ${ONETBB_MD5}
  CMAKE_CACHE_ARGS
    -DBUILD_SHARED_LIBS:BOOL=ON
)
FetchContent_GetProperties(tbb)

if(NOT tbb)
  FetchContent_Populate(tbb)
  set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON)
  set(TBB_EXAMPLES OFF)
  set(TBB_TEST OFF)
  set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)
  add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR})
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_OLD} CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(tbb)
set_property(GLOBAL PROPERTY TBB_INCLUDE_DIRS "${tbb_SOURCE_DIR}/cpp/src;${tbb_BINARY_DIR}/src")

install(TARGETS tbb tbbmalloc tbbmalloc_proxy COMPONENT OneFlowTBB)
install(DIRECTORY ${tbb_SOURCE_DIR}/include DESTINATION ${ONETBB_INSTALL_DIR} COMPONENT OneFlowTBB)

add_custom_target(install-tbb
  DEPENDS tbb tbbmalloc tbbmalloc_proxy
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_PREFIX=${ONETBB_INSTALL_DIR}
      -DCMAKE_INSTALL_COMPONENT=OneFlowTBB
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
)
