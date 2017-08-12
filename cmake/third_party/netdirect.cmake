include (ExternalProject)

set(NETDIRECT_INCLUDE_DIR ${THIRD_PARTY_DIR}/netdirect/include)

set(NETDIRECT_URL https://github.com/yuanms2/netdirect)
set(NETDIRECT_TAG 76069cbc1688a7166306375f3870415c5592ae93)
set(NETDIRECT_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/netdirect/src/netdirect/include)

ExternalProject_Add(netdirect
    PREFIX netdirect
    GIT_REPOSITORY ${NETDIRECT_URL}
    GIT_TAG ${NETDIRECT_TAG}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

if(BUILD_THIRD_PARTY)

add_custom_target(network_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${NETDIRECT_INSTALL} ${NETDIRECT_INCLUDE_DIR}
  DEPENDS netdirect)

endif(BUILD_THIRD_PARTY)