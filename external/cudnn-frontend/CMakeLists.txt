include(FetchContent)
FetchContent_Declare(
        cudnn-frontend
        URL ${CUDNN_FRONTEND_URL}
        URL_HASH MD5=${CUDNN_FRONTEND_MD5}
)
FetchContent_MakeAvailable(cudnn-frontend)
