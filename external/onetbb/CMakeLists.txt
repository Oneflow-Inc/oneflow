find_package(Threads REQUIRED)
include(FetchContent)
FetchContent_Declare(
  onetbb_external
  URL ${ONETBB_URL}
  URL_HASH MD5=${ONETBB_MD5}
)
FetchContent_GetProperties(onetbb_external)
FetchContent_MakeAvailable(onetbb_external)

set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(TBB_EXAMPLES OFF)
set(TBB_TEST OFF)
set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)

add_subdirectory(${onetbb_external_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onettb-build)
# set_property(GLOBAL PROPERTY ONETBB_SRC_DIR "${onetbb_SOURCE_DIR};${onetbb_BINARY_DIR}")
# add_custom_target(external_onetbb)
