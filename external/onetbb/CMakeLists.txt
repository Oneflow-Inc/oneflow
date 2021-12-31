find_package(Threads REQUIRED)
include(FetchContent)
FetchContent_Declare(
  onetbb
)
FetchContent_GetProperties(onetbb)

if(NOT onetbb_POPULATED)
  FetchContent_Populate(onetbb
    URL ${ONETBB_URL}
    URL_HASH MD5=${ONETBB_MD5}
  )
endif()

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


add_subdirectory(${onetbb_SOURCE_DIR} ${onetbb_BINARY_DIR})
# set_property(GLOBAL PROPERTY ONETBB_SRC_DIR "${onetbb_SOURCE_DIR};${onetbb_BINARY_DIR}")
# add_custom_target(external_onetbb)

fetchcontent_makeavailable(onetbb)