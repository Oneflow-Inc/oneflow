include(FetchContent)
FetchContent_Declare(
        cudnn_frontend
        URL ${CUDNN_FRONTEND_URL}
        URL_HASH MD5=${CUDNN_FRONTEND_MD5}
)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
FetchContent_MakeAvailable(cudnn_frontend)
