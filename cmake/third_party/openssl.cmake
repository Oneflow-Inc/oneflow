include (ExternalProject)

set(OPENSSL_INCLUDE_DIR ${THIRD_PARTY_DIR}/openssl/include)
set(OPENSSL_LIBRARY_DIR ${THIRD_PARTY_DIR}/openssl/lib)

set(OPENSSL_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/openssl/src/openssl/include)
SET(OPENSSL_TAR_URL https://github.com/openssl/openssl/archive/OpenSSL_1_1_1g.tar.gz)
set(OPENSSL_URL_HASH dd32f35dd5d543c571bc9ebb90ebe54e)
set(OPENSSL_INSTALL ${THIRD_PARTY_DIR}/openssl)
SET(OPENSSL_SOURCE_DIR ${THIRD_PARTY_SUBMODULE_DIR}/openssl)

if(THIRD_PARTY)
ExternalProject_Add(openssl
  PREFIX ${OPENSSL_SOURCE_DIR}
  URL ${OPENSSL_TAR_URL}
  URL_HASH MD5=${OPENSSL_URL_HASH}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${OPENSSL_SOURCE_DIR}/src/openssl/config --prefix=${OPENSSL_INSTALL}
  BUILD_COMMAND make -j${PROC_NUM}
  INSTALL_COMMAND make install
)
endif()
