include (ExternalProject)
include(GNUInstallDirs)

SET(ABSL_PROJECT absl)
SET(ABSL_GIT_URL https://github.com/abseil/abseil-cpp.git)
SET(ABSL_GIT_TAG 43ef2148c0936ebf7cb4be6b19927a9d9d145b8f)
 
SET(ABSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/absl/src/absl)
SET(ABSL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/absl/install)

SET(ABSL_INCLUDE_DIR ${THIRD_PARTY_DIR}/absl/include CACHE PATH "" FORCE)
SET(ABSL_LIBRARY_DIR ${THIRD_PARTY_DIR}/absl/lib CACHE PATH "" FORCE)

SET(ABSL_LIBRARIES
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_base.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_spinlock_wait.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_dynamic_annotations.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_malloc_internal.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_throw_delegate.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_int128.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_strings.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_str_format_internal.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_time.a
    ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}/libabsl_bad_optional_access.a)

if (THIRD_PARTY)
  ExternalProject_Add(${ABSL_PROJECT}
    PREFIX absl 
    GIT_REPOSITORY ${ABSL_GIT_URL}
    GIT_TAG ${ABSL_GIT_TAG}
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

add_custom_target(absl_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ABSL_LIBRARY_DIR}
  DEPENDS absl)

add_custom_target(absl_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${ABSL_INSTALL}/include ${ABSL_INCLUDE_DIR}
  DEPENDS absl_create_library_dir)

add_custom_target(absl_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ABSL_LIBRARIES} ${ABSL_LIBRARY_DIR}
  DEPENDS absl_create_library_dir)
endif(THIRD_PARTY)
