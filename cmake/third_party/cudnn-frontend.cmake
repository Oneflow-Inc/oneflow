include(ExternalProject)

set(CUDNN_FRONTEND_URL https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.1.2.zip)
set(CUDNN_FRONTEND_MD5 7e16cc2dcaddefa7fd0f3d82b9cf5d73)
use_mirror(VARIABLE CUDNN_FRONTEND_URL URL ${CUDNN_FRONTEND_URL})

set(CUDNN_FRONTEND_INCLUDE_DIR ${THIRD_PARTY_DIR}/cudnn-frontend/include)
set(CUDNN_FRONTEND_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudnn-frontend/src/cudnn-frontend)

if(THIRD_PARTY)
  ExternalProject_Add(
    cudnn-frontend
    PREFIX cudnn-frontend
    URL ${CUDNN_FRONTEND_URL}
    URL_MD5 ${CUDNN_FRONTEND_MD5}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")

  add_copy_headers_target(
    NAME
    cudnn_frontend
    SRC
    ${CUDNN_FRONTEND_BASE_DIR}/include/
    DST
    ${CUDNN_FRONTEND_INCLUDE_DIR}
    DEPS
    cudnn-frontend)
endif(THIRD_PARTY)
