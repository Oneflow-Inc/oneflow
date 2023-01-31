include(ExternalProject)

set_mirror_url_with_hash(
  glog_URL https://github.com/google/glog/archive/8f9ccfe770add9e4c64e9b25c102658e3c763b73.tar.gz
  b2d2becff6d7d5577a771180ab7da617)

include(FetchContent)

FetchContent_Declare(glog URL ${glog_URL} URL_HASH MD5=${glog_URL_HASH})

set(WITH_GFLAGS OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(WITH_GTEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(glog)

# just for tensorflow, DO NOT USE IN OTHER PLACE
FetchContent_GetProperties(glog)
set(GLOG_INCLUDE_DIR ${glog_BINARY_DIR})
