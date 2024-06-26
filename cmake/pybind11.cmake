include(FetchContent)

set_mirror_url_with_hash(PYBIND11_URL https://github.com/pybind/pybind11/archive/v2.11.1.zip
                         c62d9e05243bd31cdb3bae1bb2f56655)

FetchContent_Declare(pybind11 URL ${PYBIND11_URL} URL_HASH MD5=${PYBIND11_URL_HASH})

FetchContent_MakeAvailable(pybind11)
