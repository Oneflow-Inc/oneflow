include(ExternalProject)
include(GNUInstallDirs)

SET(ABSL_PROJECT absl)
SET(ABSL_TAR_URL https://github.com/Oneflow-Inc/abseil-cpp/archive/d0.tar.gz)
use_mirror(VARIABLE ABSL_TAR_URL URL ${ABSL_TAR_URL})
SET(ABSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/absl/src/absl)
SET(ABSL_INSTALL ${THIRD_PARTY_DIR}/absl)

SET(ABSL_INCLUDE_DIR ${THIRD_PARTY_DIR}/absl/include CACHE PATH "" FORCE)
SET(ABSL_LIBRARY_DIR ${THIRD_PARTY_DIR}/absl/${CMAKE_INSTALL_LIBDIR} CACHE PATH "" FORCE)

if(WIN32)
  set(ABSL_BUILD_LIBRARY_DIR ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  set(ABSL_LIBRARY_NAMES absl_base.lib absl_spinlock_wait.lib absl_dynamic_annotations.lib
    absl_malloc_internal.lib absl_throw_delegate.lib absl_int128.lib absl_strings.lib absl_str_format_internal.lib
    absl_time.lib absl_bad_optional_access.lib)
else()
  set(ABSL_BUILD_LIBRARY_DIR ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  set(ABSL_LIBRARY_NAMES libabsl_base.a libabsl_spinlock_wait.a libabsl_dynamic_annotations.a
    libabsl_malloc_internal.a libabsl_throw_delegate.a libabsl_int128.a libabsl_strings.a libabsl_str_format_internal.a
    libabsl_time.a libabsl_bad_optional_access.a)
endif()

foreach(LIBRARY_NAME ${ABSL_LIBRARY_NAMES})
  list(APPEND ABSL_STATIC_LIBRARIES ${ABSL_LIBRARY_DIR}/${LIBRARY_NAME})
  list(APPEND ABSL_BUILD_STATIC_LIBRARIES ${ABSL_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)
  ExternalProject_Add(${ABSL_PROJECT}
    PREFIX absl
    URL ${ABSL_TAR_URL} 
    URL_MD5 20126998c9b17e5f7a93711972f03f79
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
    CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${ABSL_INSTALL}
        -DCMAKE_INSTALL_LIBDIR:PATH=${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  )
endif(THIRD_PARTY)
