include(ExternalProject)
set(CARES_TAR_URL
    https://github.com/c-ares/c-ares/releases/download/cares-1_15_0/c-ares-1.15.0.tar.gz)
use_mirror(VARIABLE CARES_TAR_URL URL ${CARES_TAR_URL})
set(CARES_URL_HASH d2391da274653f7643270623e822dff7)
set(CARES_INSTALL ${THIRD_PARTY_DIR}/cares)
set(CARES_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cares/src/cares)

if(THIRD_PARTY)
  ExternalProject_Add(
    cares
    PREFIX cares
    URL ${CARES_TAR_URL}
    URL_HASH MD5=${CARES_URL_HASH}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")

endif()
