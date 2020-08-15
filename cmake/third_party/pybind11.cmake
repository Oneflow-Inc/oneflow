include(FetchContent)

if(THIRD_PARTY)
    FetchContent_Declare(
        pybind11
        URL https://github.com/Oneflow-Inc/pybind11/archive/1534e17.tar.gz
    )
    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
endif(THIRD_PARTY)