include (ExternalProject)

set(OPENSSL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/openssl/install)

SET(OPENSSL_TAR_URL https://github.com/openssl/openssl/archive/OpenSSL_1_1_1g.tar.gz)
set(OPENSSL_URL_HASH dd32f35dd5d543c571bc9ebb90ebe54e)
set (OPENSSL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openssl)

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
endif()
