include(FetchContent)

set_mirror_url_with_hash(JSON_URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip
                         49097a7ec390ffaf1cd2e14b734b6c75)
set(JSON_Install ON CACHE STRING "" FORCE)

FetchContent_Declare(json URL ${JSON_URL} URL_HASH MD5=${JSON_URL_HASH})

FetchContent_MakeAvailable(json)
