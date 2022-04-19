include(ExternalProject)

if(UNIX AND NOT APPLE)
  set(BUILD_HWLOC_DEFAULT ON)
else()
  set(BUILD_HWLOC_DEFAULT OFF)
endif()
option(BUILD_HWLOC "" ${BUILD_HWLOC_DEFAULT})

if(BUILD_HWLOC)

  set(PCIACCESS_INSTALL ${THIRD_PARTY_DIR}/pciaccess)
  set(PCIACCESS_INCLUDE_DIR ${PCIACCESS_INSTALL}/include)
  set(PCIACCESS_LIBRARY_DIR ${PCIACCESS_INSTALL}/lib)
  set(PCIACCESS_LIBRARY_NAMES libpciaccess.a)
  foreach(LIBRARY_NAME ${PCIACCESS_LIBRARY_NAMES})
    list(APPEND PCIACCESS_STATIC_LIBRARIES ${PCIACCESS_LIBRARY_DIR}/${LIBRARY_NAME})
  endforeach()

  set(HWLOC_INSTALL ${THIRD_PARTY_DIR}/hwloc)
  set(HWLOC_INCLUDE_DIR ${HWLOC_INSTALL}/include)
  set(HWLOC_LIBRARY_DIR ${HWLOC_INSTALL}/lib)
  set(HWLOC_LIBRARY_NAMES libhwloc.a)
  foreach(LIBRARY_NAME ${HWLOC_LIBRARY_NAMES})
    list(APPEND ONEFLOW_HWLOC_STATIC_LIBRARIES ${HWLOC_LIBRARY_DIR}/${LIBRARY_NAME})
  endforeach()

  if(THIRD_PARTY)

    include(ProcessorCount)
    ProcessorCount(PROC_NUM)

    set(XORG_MACROS_INSTALL ${THIRD_PARTY_DIR}/xorg-macros)
    set(XORG_MACROS_TAR_URL
        https://github.com/freedesktop/xorg-macros/archive/refs/tags/util-macros-1.19.1.tar.gz)
    use_mirror(VARIABLE XORG_MACROS_TAR_URL URL ${XORG_MACROS_TAR_URL})
    set(XORG_MACROS_URL_HASH 37afda9e9b44ecb9b2c16293bacd0e21)
    set(XORG_MACROS_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/xorg-macros)
    set(XORG_MACROS_PKG_CONFIG_DIR ${XORG_MACROS_INSTALL}/share/pkgconfig)

    ExternalProject_Add(
      xorg-macros
      PREFIX xorg-macros
      URL ${XORG_MACROS_TAR_URL}
      URL_HASH MD5=${XORG_MACROS_URL_HASH}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${XORG_MACROS_SOURCE_DIR}/src/xorg-macros/autogen.sh
      COMMAND ${XORG_MACROS_SOURCE_DIR}/src/xorg-macros/configure --prefix=${XORG_MACROS_INSTALL}
      BUILD_COMMAND make -j${PROC_NUM}
      INSTALL_COMMAND make install)

    set(PCIACCESS_TAR_URL
        https://github.com/freedesktop/xorg-libpciaccess/archive/refs/tags/libpciaccess-0.16.tar.gz)
    use_mirror(VARIABLE PCIACCESS_TAR_URL URL ${PCIACCESS_TAR_URL})
    set(PCIACCESS_URL_HASH 92e2b604e294a9160bc977c000507340)
    set(PCIACCESS_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pciaccess)

    set(PCIACCESS_CFLAGS "-O3 -fPIC")

    ExternalProject_Add(
      pciaccess
      PREFIX pciaccess
      URL ${PCIACCESS_TAR_URL}
      URL_HASH MD5=${PCIACCESS_URL_HASH}
      UPDATE_COMMAND ""
      PATCH_COMMAND cp ${XORG_MACROS_INSTALL}/share/aclocal/xorg-macros.m4
                    ${PCIACCESS_SOURCE_DIR}/src/pciaccess/m4
      CONFIGURE_COMMAND ${PCIACCESS_SOURCE_DIR}/src/pciaccess/autogen.sh
      COMMAND ${PCIACCESS_SOURCE_DIR}/src/pciaccess/configure --prefix=${PCIACCESS_INSTALL}
              --enable-shared=no
      BUILD_COMMAND make -j${PROC_NUM} CFLAGS=${PCIACCESS_CFLAGS}
      BUILD_BYPRODUCTS ${PCIACCESS_STATIC_LIBRARIES}
      INSTALL_COMMAND make install
      DEPENDS xorg-macros)
    set(HWLOC_TAR_URL https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.4.1.tar.gz)
    use_mirror(VARIABLE HWLOC_TAR_URL URL ${HWLOC_TAR_URL})
    set(HWLOC_URL_HASH ac25fc7c2a665b7914c6c21b782f1c4f)
    set(HWLOC_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hwloc)

    set(HWLOC_CFLAGS "-O3 -fPIC")

    ExternalProject_Add(
      hwloc
      PREFIX hwloc
      URL ${HWLOC_TAR_URL}
      URL_HASH MD5=${HWLOC_URL_HASH}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${HWLOC_SOURCE_DIR}/src/hwloc/autogen.sh
      COMMAND ${HWLOC_SOURCE_DIR}/src/hwloc/configure --prefix=${HWLOC_INSTALL}
              PKG_CONFIG_PATH=${PCIACCESS_INSTALL}/lib/pkgconfig --disable-libxml2 --enable-static
              --enable-shared=no
      BUILD_COMMAND make -j${PROC_NUM} CFLAGS=${HWLOC_CFLAGS}
      BUILD_BYPRODUCTS ${ONEFLOW_HWLOC_STATIC_LIBRARIES}
      INSTALL_COMMAND make install
      DEPENDS pciaccess)
  endif(THIRD_PARTY)

endif(BUILD_HWLOC)
