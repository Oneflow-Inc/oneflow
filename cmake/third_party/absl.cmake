include(ExternalProject)
include(GNUInstallDirs)

set(ABSL_PROJECT absl)
set(ABSL_TAR_URL
    https://github.com/abseil/abseil-cpp/archive/78be63686ba732b25052be15f8d6dee891c05749.tar.gz)
use_mirror(VARIABLE ABSL_TAR_URL URL ${ABSL_TAR_URL})
set(ABSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/absl/src/absl)
set(ABSL_INSTALL ${THIRD_PARTY_DIR}/absl)

set(ABSL_INCLUDE_DIR ${THIRD_PARTY_DIR}/absl/include CACHE PATH "" FORCE)
set(ABSL_LIBRARY_DIR ${THIRD_PARTY_DIR}/absl/${CMAKE_INSTALL_LIBDIR} CACHE PATH "" FORCE)

if(WIN32)
  set(ABSL_BUILD_LIBRARY_DIR ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  set(ABSL_LIBRARY_NAMES
      absl_spinlock_wait.lib
      absl_dynamic_annotations.lib
      absl_malloc_internal.lib
      absl_throw_delegate.lib
      absl_int128.lib
      absl_strings.lib
      absl_str_format_internal.lib
      absl_time.lib
      absl_bad_optional_access.lib
      absl_base.lib)
else()
  set(ABSL_BUILD_LIBRARY_DIR ${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  set(ABSL_LIBRARY_NAMES
      libabsl_spinlock_wait.a
      libabsl_malloc_internal.a
      libabsl_throw_delegate.a
      libabsl_int128.a
      libabsl_strings.a
      libabsl_str_format_internal.a
      libabsl_time.a
      libabsl_bad_optional_access.a
      libabsl_base.a
      libabsl_status.a
      libabsl_synchronization.a
      libabsl_cord.a
      libabsl_cord_internal.a
      libabsl_cordz_handle.a
      libabsl_cordz_functions.a
      libabsl_cordz_info.a
      libabsl_raw_logging_internal.a
      libabsl_strings_internal.a
      libabsl_crc32c.a
      libabsl_crc_cord_state.a
      libabsl_crc_internal.a
      libabsl_random_internal_pool_urbg.a
      libabsl_random_internal_seed_material.a
      libabsl_random_internal_platform.a
      libabsl_random_internal_randen.a
      libabsl_random_internal_randen_hwaes.a
      libabsl_random_internal_randen_hwaes_impl.a
      libabsl_random_internal_randen_slow.a
      libabsl_random_seed_gen_exception.a
      libabsl_bad_variant_access.a
      libabsl_time_zone.a
      libabsl_exponential_biased.a
      libabsl_statusor.a
      libabsl_strerror.a
      libabsl_hash.a
      libabsl_low_level_hash.a
      libabsl_city.a
      libabsl_symbolize.a
      libabsl_debugging_internal.a
      libabsl_demangle_internal.a
      libabsl_stacktrace.a)
endif()

foreach(LIBRARY_NAME ${ABSL_LIBRARY_NAMES})
  list(APPEND ABSL_STATIC_LIBRARIES ${ABSL_LIBRARY_DIR}/${LIBRARY_NAME})
  list(APPEND ABSL_BUILD_STATIC_LIBRARIES ${ABSL_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)
  ExternalProject_Add(
    ${ABSL_PROJECT}
    PREFIX absl
    URL ${ABSL_TAR_URL}
    URL_MD5 3db4fcf154b69ff52b2f43e583d89ded
    UPDATE_COMMAND ""
    BUILD_BYPRODUCTS ${ABSL_STATIC_LIBRARIES}
    CMAKE_CACHE_ARGS
      -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_INSTALL_PREFIX:PATH=${ABSL_INSTALL}
      -DCMAKE_INSTALL_LIBDIR:PATH=${ABSL_INSTALL}/${CMAKE_INSTALL_LIBDIR}
      -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD})
endif(THIRD_PARTY)
