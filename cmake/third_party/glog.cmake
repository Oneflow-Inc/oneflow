include(ExternalProject)

set_mirror_url_with_hash(glog_URL https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
                         2368e3e0a95cce8b5b35a133271b480f)

include(FetchContent)

FetchContent_Declare(glog URL ${glog_URL} URL_HASH MD5=${glog_URL_HASH})

set(WITH_GFLAGS OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(WITH_GTEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(glog)

# just for tensorflow, DO NOT USE IN OTHER PLACE
FetchContent_GetProperties(glog)
set(GLOG_INCLUDE_DIR ${glog_BINARY_DIR})
