include (ExternalProject)

set_mirror_url_with_hash(glog_URL 
  https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
  2368e3e0a95cce8b5b35a133271b480f
)

include(FetchContent)

FetchContent_Declare(
  glog
  URL ${glog_URL}
  URL_HASH MD5=${glog_URL_HASH}
)

# make a function to intro a scope for variables
function(FetchContent_Glog)
  set(WITH_GFLAGS OFF)
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(glog)
endfunction()

FetchContent_Glog()

# just for tensorflow, DO NOT USE IN OTHER PLACE
FetchContent_GetProperties(glog)
set(GLOG_INCLUDE_DIR ${glog_BINARY_DIR})
