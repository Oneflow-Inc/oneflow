include(FetchContent)
FetchContent_Declare(
        robin_hood_hashing
)
FetchContent_GetProperties(robin_hood_hashing)

if(NOT robin_hood_hashing_POPULATED)
    FetchContent_Populate(robin_hood_hashing
            URL ${ROBIN_HOOD_HASHING_URL}
            URL_HASH MD5=${ROBIN_HOOD_HASHING_MD5}
            )
endif()

set_property(GLOBAL PROPERTY ROBIN_HOOD_HASHING_INCLUDE_DIR "${robin_hood_hashing_SOURCE_DIR}/src/include")
