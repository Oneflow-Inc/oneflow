include(FetchContent)

# v3.10.4
set_mirror_url_with_hash(JSON_URL https://github.com/nlohmann/json/archive/fec56a1a16c6e1c1b1f4e116a20e79398282626c.zip
                         4b2eb562b0fedd37e88602fc4fb76b6b)
set(JSON_Install ON CACHE STRING "" FORCE)

FetchContent_Declare(json URL ${JSON_URL} URL_HASH MD5=${JSON_URL_HASH})

FetchContent_MakeAvailable(json)
