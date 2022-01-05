include(FetchContent)

set_mirror_url_with_hash(JSON_URL 
  https://github.com/nlohmann/json/archive/refs/tags/v3.10.4.zip
  59c2a25e17b94d612fdb32a1a37378cf
)
set(JSON_Install ON CACHE STRING "" FORCE)

FetchContent_Declare(
    json
    URL ${JSON_URL}
    URL_HASH MD5=${JSON_URL_HASH}
)


FetchContent_MakeAvailable(json)
