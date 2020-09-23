include(FetchContent)
set(PYBIND11_TAR_URL https://github.com/Oneflow-Inc/pybind11/archive/1534e17.zip)
use_mirror(NAME PYBIND11_TAR_URL URL ${PYBIND11_TAR_URL})

FetchContent_Declare(
    pybind11
    URL ${PYBIND11_TAR_URL} 
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
