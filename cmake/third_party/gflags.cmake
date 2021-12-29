include (ExternalProject)

set(gflags_URL https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz)
use_mirror(VARIABLE gflags_URL URL ${gflags_URL})
set(gflags_URL_HASH 1a865b93bacfa963201af3f75b7bd64c)

include(FetchContent)

set(GFLAGS_INSTALL_HEADERS ON)
set(GFLAGS_INSTALL_STATIC_LIBS ON)
set(GFLAGS_NAMESPACE gflags)

FetchContent_Declare(
    gflags
    URL ${gflags_URL}
    URL_HASH MD5=${gflags_URL_HASH}

    # refer to https://github.com/gflags/gflags/issues/306
    PATCH_COMMAND git apply "${CMAKE_CURRENT_LIST_DIR}/patches/gflags-v2.2.2.patch"
)

if (THIRD_PARTY)
    FetchContent_MakeAvailable(gflags)
endif(THIRD_PARTY)
