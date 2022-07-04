include(FetchContent)

set(FMT_INSTALL_DIR ${THIRD_PARTY_DIR}/fmt)

FetchContent_Declare(fmt URL ${FMT_URL} URL_HASH MD5=${FMT_MD5})

FetchContent_MakeAvailable(fmt)

install(
  TARGETS fmt
  EXPORT oneflow
  LIBRARY DESTINATION ${FMT_INSTALL_DIR}/lib
  ARCHIVE DESTINATION ${FMT_INSTALL_DIR}/lib)
install(DIRECTORY ${fmt_SOURCE_DIR}/include DESTINATION ${FMT_INSTALL_DIR})
