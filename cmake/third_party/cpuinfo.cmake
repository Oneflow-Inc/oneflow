include (ExternalProject)

set(CPUINFO_INSTALL_DIR ${THIRD_PARTY_DIR}/cpuinfo)
set(CPUINFO_INCLUDE_DIR ${CPUINFO_INSTALL_DIR}/include)
set(CPUINFO_LIBRARY_DIR ${CPUINFO_INSTALL_DIR}/lib)

set(cpuinfo_URL https://github.com/Oneflow-Inc/cpuinfo/archive/v0.1.0.tar.gz)
use_mirror(VARIABLE cpuinfo_URL URL ${cpuinfo_URL})

if(WIN32)
    set(CPUINFO_LIBRARY_NAMES cpuinfo.lib)
else()
    if(BUILD_SHARED_LIBS)
      if("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".dylib")
        set(CPUINFO_LIBRARY_NAMES libcpuinfo.dylib)
      elseif("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".so")
        set(CPUINFO_LIBRARY_NAMES libcpuinfo.so)
      else()
        message(FATAL_ERROR "${CMAKE_SHARED_LIBRARY_SUFFIX} not support for cpuinfo")
      endif()
    else()
      set(CPUINFO_LIBRARY_NAMES libcpuinfo.a libclog.a)
    endif()
endif()

foreach(LIBRARY_NAME ${CPUINFO_LIBRARY_NAMES})
    list(APPEND CPUINFO_STATIC_LIBRARIES ${CPUINFO_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set (CPUINFO_PUBLIC_H
  ${CMAKE_CURRENT_BINARY_DIR}/cpuinfo/include/cpuinfo.h
  ${CMAKE_CURRENT_BINARY_DIR}/cpuinfo/include/clog.h
)

if(THIRD_PARTY)

ExternalProject_Add(libcpuinfo
    PREFIX libcpuinfo
    URL ${cpuinfo_URL}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${CPUINFO_STATIC_LIBRARIES}
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
        -DCMAKE_INSTALL_PREFIX:STRING=${CPUINFO_INSTALL_DIR}
)

add_custom_target(libcpuinfo_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENCV_LIBRARY_DIR}
  DEPENDS libcpuinfo)

endif(THIRD_PARTY)
