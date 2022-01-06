include (ExternalProject)

set(GLOG_INSTALL_DIR ${THIRD_PARTY_DIR}/glog/install)
set(GLOG_INCLUDE_DIR ${GLOG_INSTALL_DIR}/include)
set(GLOG_LIBRARY_DIR ${GLOG_INSTALL_DIR}/lib)

set(glog_URL https://github.com/Oneflow-Inc/glog/archive/4f3e18bf2.tar.gz)
use_mirror(VARIABLE glog_URL URL ${glog_URL})

if(WIN32)
    set(GLOG_LIBRARY_NAMES glog.lib)
else()
    if(BUILD_SHARED_LIBS)
      if("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".dylib")
        set(GLOG_LIBRARY_NAMES libglog.dylib)
      elseif("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".so")
        set(GLOG_LIBRARY_NAMES libglog.so)
      else()
        message(FATAL_ERROR "${CMAKE_SHARED_LIBRARY_SUFFIX} not support for glog")
      endif()
    else()
      set(GLOG_LIBRARY_NAMES libglog.a)
    endif()
endif()

foreach(LIBRARY_NAME ${GLOG_LIBRARY_NAMES})
    list(APPEND GLOG_STATIC_LIBRARIES ${GLOG_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set (GLOG_PUBLIC_H
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/config.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/raw_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/stl_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/vlog_is_on.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/src/glog/log_severity.h
)

if(THIRD_PARTY)

ExternalProject_Add(glog
    DEPENDS gflags
    PREFIX glog
    URL ${glog_URL}
    URL_MD5 3ca928ef755c0a890680e023e3d4b9a6
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${GLOG_STATIC_LIBRARIES}
    CMAKE_CACHE_ARGS
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_POLICY_DEFAULT_CMP0074:STRING=NEW
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
        -DBUILD_SHARED_LIBS:BOOL=${PROTOBUF_BUILD_SHARED_LIBS}
        -DBUILD_TESTING:BOOL=OFF
        -DWITH_GFLAGS:BOOL=ON
        -Dgflags_ROOT:STRING=${GFLAGS_INSTALL_DIR}
        -DCMAKE_INSTALL_PREFIX:STRING=${GLOG_INSTALL_DIR}
        -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
)

endif(THIRD_PARTY)
add_library(glog_imported UNKNOWN IMPORTED)
set_property(TARGET glog_imported PROPERTY IMPORTED_LOCATION "${GLOG_STATIC_LIBRARIES}")
