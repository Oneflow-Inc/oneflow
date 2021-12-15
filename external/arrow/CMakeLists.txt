find_package(Threads REQUIRED)
include(FetchContent)
FetchContent_Declare(
  arrow
)
FetchContent_GetProperties(arrow)

if(NOT arrow_POPULATED)
  FetchContent_Populate(arrow
    URL ${ARROW_URL}
    URL_HASH MD5=${ARROW_MD5}
  )
endif()
set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)
set(ARROW_CMAKE_INSTALL_DIR ${THIRD_PARTY_DIR}/arrow)
set(ARROW_BUILD_TESTS OFF)
set(ARROW_PARQUET ON)
set(PARQUET_MINIMAL_DEPENDENCY ON)
include(CMakePackageConfigHelpers)
function(install)
    message(STATUS "skip installing ${ARGV}")
endfunction()
add_subdirectory(${arrow_SOURCE_DIR}/cpp ${arrow_BINARY_DIR})

# include(${arrow_SOURCE_DIR}/cpp/cmake_modules/BuildUtils.cmake)
# set(ARROW_VERSION "7.0.0-SNAPSHOT-oneflow")
# string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" ARROW_BASE_VERSION "${ARROW_VERSION}")
# project(external_arrow VERSION "${ARROW_BASE_VERSION}")
# message(STATUS "arrow_SOURCE_DIR: ${arrow_SOURCE_DIR}")
# add_subdirectory(${arrow_SOURCE_DIR}/cpp/src/parquet ${PROJECT_BINARY_DIR}/parquet)
# include_directories(${arrow_BINARY_DIR})
