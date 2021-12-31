include(FetchContent)

set(PYBIND11_URL https://github.com/pybind/pybind11/archive/v2.7.0.zip)
use_mirror(VARIABLE PYBIND11_URL URL ${PYBIND11_URL})
set(PYBIND11_URL_HASH 267807f790ef598ef912a79aceefdc10)

FetchContent_Declare(
    pybind11
    URL ${PYBIND11_URL} 
    URL_HASH MD5=${PYBIND11_URL_HASH}
)

FetchContent_MakeAvailable(pybind11)
