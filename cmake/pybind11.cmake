include(FetchContent)
set(PYBIND11_TAR_URL https://github.com/pybind/pybind11/archive/v2.6.0.zip)
use_mirror(VARIABLE PYBIND11_TAR_URL URL ${PYBIND11_TAR_URL})
set(PYBIND11_URL_HASH 7d7d926f8b00fb181dd8aeea0451dbc3)

FetchContent_Declare(
    pybind11
    URL ${PYBIND11_TAR_URL} 
    URL_HASH MD5=${PYBIND11_URL_HASH}
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    include_directories("${pybind11_SOURCE_DIR}/include")
endif()
