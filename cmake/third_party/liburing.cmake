include(ExternalProject)

option(WITH_LIBURING "" OFF)

if(WITH_LIBURING)

  set(LIBURING_INSTALL ${THIRD_PARTY_DIR}/liburing)
  set(LIBURING_INCLUDE_DIR ${LIBURING_INSTALL}/include)
  set(LIBURING_LIBRARY_DIR ${LIBURING_INSTALL}/lib)
  set(LIBURING_LIBRARY_NAMES liburing.a)
  foreach(LIBRARY_NAME ${LIBURING_LIBRARY_NAMES})
    list(APPEND LIBURING_STATIC_LIBRARIES ${LIBURING_LIBRARY_DIR}/${LIBRARY_NAME})
  endforeach()

  if(THIRD_PARTY)

    include(ProcessorCount)
    ProcessorCount(PROC_NUM)

    set(LIBURING_URL https://github.com/axboe/liburing/archive/refs/tags/liburing-2.1.tar.gz)
    use_mirror(VARIABLE LIBURING_URL URL ${LIBURING_URL})
    set(LIBURING_URL_HASH 78f13d9861b334b9a9ca0d12cf2a6d3c)
    set(LIBURING_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/liburing)

    set(LIBURING_CFLAGS "-O3 -fPIC")

    ExternalProject_Add(
      liburing
      PREFIX liburing
      URL ${LIBURING_URL}
      URL_HASH MD5=${LIBURING_URL_HASH}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND cd ${LIBURING_SOURCE_DIR}/src/liburing/ && ./configure
                        --prefix=${LIBURING_INSTALL}
      BUILD_COMMAND cd ${LIBURING_SOURCE_DIR}/src/liburing/ && make -j${PROC_NUM}
                    CFLAGS=${LIBURING_CFLAGS} install
      BUILD_BYPRODUCTS ${LIBURING_STATIC_LIBRARIES}
      INSTALL_COMMAND "")

  endif(THIRD_PARTY)

endif(WITH_LIBURING)
