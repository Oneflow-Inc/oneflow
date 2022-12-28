include(FetchContent)
FetchContent_Declare(
        robin_hood_hashing
        URL ${ROBIN_HOOD_HASHING_URL}
        URL_HASH MD5=${ROBIN_HOOD_HASHING_MD5}
)
FetchContent_MakeAvailable(robin_hood_hashing)
