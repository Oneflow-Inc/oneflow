include (ExternalProject)

set(ONEDNN_INSTALL_DIR ${THIRD_PARTY_DIR}/onednn)
set(ONEDNN_INCLUDE_DIR ${ONEDNN_INSTALL_DIR}/include)
set(ONEDNN_LIBRARY_DIR ${ONEDNN_INSTALL_DIR}/lib)

set(oneDNN_URL https://github.com/oneapi-src/oneDNN/archive/refs/tags/v2.4.3.tar.gz)
use_mirror(VARIABLE oneDNN_URL URL ${oneDNN_URL})

if(WIN32)
    # set(ONEDNN_LIBRARY_NAMES onednn.lib)
else()
    if(BUILD_SHARED_LIBS)
      if("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".dylib")
        set(ONEDNN_LIBRARY_NAMES libdnnl.dylib)
      elseif("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".so")
        set(ONEDNN_LIBRARY_NAMES libdnnl.so)
        set(DNNL_LIBRARY_TYPE SHARED)
      else()
        message(FATAL_ERROR "${CMAKE_SHARED_LIBRARY_SUFFIX} not support for onednn")
      endif()
    else()
      set(ONEDNN_LIBRARY_NAMES libdnnl.a )
      set(DNNL_LIBRARY_TYPE STATIC)
    endif()
endif()

foreach(LIBRARY_NAME ${ONEDNN_LIBRARY_NAMES})
    list(APPEND ONEDNN_STATIC_LIBRARIES ${ONEDNN_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()


if(THIRD_PARTY)

ExternalProject_Add(onednn
    PREFIX onednn
    URL ${oneDNN_URL}
    URL_MD5 c60ea96acbaccec053be7e3fa81c6184
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE OFF
    BUILD_BYPRODUCTS ${ONEDNN_STATIC_LIBRARIES}
    CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:STRING=${ONEDNN_INSTALL_DIR}
        -DDNNL_IS_MAIN_PROJECT:BOOL=OFF
        -DDNNL_BUILD_EXAMPLES:BOOL=OFF
        -DDNNL_BUILD_TESTS:BOOL=OFF
        -DDNNL_LIBRARY_TYPE:STRING=${DNNL_LIBRARY_TYPE}
        -DDNNL_CPU_RUNTIME=OMP
)


endif(THIRD_PARTY)
