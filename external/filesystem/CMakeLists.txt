include(FetchContent)

FetchContent_Declare(filesystem URL ${FILESYSTEM_URL} URL_HASH MD5=${FILESYSTEM_MD5})

FetchContent_MakeAvailable(filesystem)