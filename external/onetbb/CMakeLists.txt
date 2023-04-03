find_package(Threads REQUIRED)
set(ONETBB_INSTALL_DIR ${THIRD_PARTY_DIR}/tbb CACHE PATH " ")

include(FetchContent)
FetchContent_Declare(tbb URL ${ONETBB_URL} URL_HASH MD5=${ONETBB_MD5})
FetchContent_GetProperties(tbb)

set(TBB_EXAMPLES OFF CACHE BOOL "")
set(TBB_TEST OFF CACHE BOOL "")
set(TBB_ENABLE_IPO OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)

FetchContent_MakeAvailable(tbb)

set(TBBBIND_LIBRARY_NAME)
if(HWLOC_VERSION)
  if(HWLOC_VERSION VERSION_LESS 2)
    set(TBBBIND_LIBRARY_NAME tbbbind)
  elseif(HWLOC_VERSION VERSION_LESS 2.5)
    set(TBBBIND_LIBRARY_NAME tbbbind_2_0)
  else()
    set(TBBBIND_LIBRARY_NAME tbbbind_2_5)
  endif()
endif()

add_custom_target(
  install-tbb DEPENDS tbb tbbmalloc tbbmalloc_proxy ${TBBBIND_LIBRARY_NAME}
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_PREFIX=${ONETBB_INSTALL_DIR} -P
          "${tbb_BINARY_DIR}/cmake_install.cmake")
