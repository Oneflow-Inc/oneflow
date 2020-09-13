include (ExternalProject)

set(OPENSSL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/openssl/install)
set(OPENSSL_INCLUDE_DIR ${THIRD_PARTY_DIR}/openssl/include)
set(OPENSSL_LIBRARY_DIR ${THIRD_PARTY_DIR}/openssl/lib)

SET(OPENSSL_TAR_URL https://github.com/openssl/openssl/archive/OpenSSL_1_1_1g.tar.gz)
set(OPENSSL_URL_HASH dd32f35dd5d543c571bc9ebb90ebe54e)
set (OPENSSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openssl)

if(WIN32)
  set(OPENSSL_BUILD_LIBRARY_DIR ${OPENSSL_INSTALL}/lib)
  set(OPENSSL_LIBRARY_NAMES ssl.lib crypto.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
  set(OPENSSL_BUILD_LIBRARY_DIR ${OPENSSL_INSTALL}/lib)
  set(OPENSSL_LIBRARY_NAMES libssl.a libcrypto.a)
else()
  set(OPENSSL_BUILD_LIBRARY_DIR ${OPENSSL_INSTALL}/lib)
  set(OPENSSL_LIBRARY_NAMES libssl.a libcrypto.a)
endif()

foreach(LIBRARY_NAME ${OPENSSL_LIBRARY_NAMES})
  list(APPEND OPENSSL_STATIC_LIBRARIES ${OPENSSL_LIBRARY_DIR}/${LIBRARY_NAME})
  list(APPEND OPENSSL_BUILD_STATIC_LIBRARIES ${OPENSSL_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)
ExternalProject_Add(openssl
  PREFIX openssl 
  URL ${OPENSSL_TAR_URL}
  URL_HASH MD5=${OPENSSL_URL_HASH}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${OPENSSL_SOURCE_DIR}/src/openssl/config --prefix=${OPENSSL_INSTALL}
  BUILD_COMMAND make -j${PROC_NUM}
  INSTALL_COMMAND make install
)

add_custom_target(openssl_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENSSL_LIBRARY_DIR}
  DEPENDS openssl)

add_custom_target(openssl_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${OPENSSL_INSTALL}/include ${OPENSSL_INCLUDE_DIR}
  DEPENDS openssl_create_library_dir)

add_custom_target(openssl_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENSSL_BUILD_STATIC_LIBRARIES} ${OPENSSL_LIBRARY_DIR}
  DEPENDS openssl_create_library_dir)
endif(THIRD_PARTY)
