include(FetchContent)
FetchContent_Declare(
        cudnn_frontend
        URL ${CUDNN_FRONTEND_URL}
        URL_HASH MD5=${CUDNN_FRONTEND_MD5}
)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
FetchContent_MakeAvailable(cudnn_frontend)

set(CUDNN_FRONTEND_INSTALL_DIR ${THIRD_PARTY_DIR}/cudnn_frontend)
install(
  TARGETS cudnn_frontend
  EXPORT oneflow
  LIBRARY DESTINATION ${CUDNN_FRONTEND_INSTALL_DIR}/lib
  ARCHIVE DESTINATION ${CUDNN_FRONTEND_INSTALL_DIR}/lib
)
