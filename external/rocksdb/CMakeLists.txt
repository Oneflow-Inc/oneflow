find_package(Threads REQUIRED)
include(FetchContent)
FetchContent_Declare(
        rocksdb
)
FetchContent_GetProperties(rocksdb)

if(NOT rocksdb_POPULATED)
    FetchContent_Populate(rocksdb
            URL ${ROCKSDB_URL}
            URL_HASH MD5=${ROCKSDB_MD5}
            )
endif()

set(WITH_TESTS "" CACHE BOOL OFF FORCE)
set(WITH_BENCHMARK_TOOLS "" CACHE BOOL OFF FORCE)
set(WITH_CORE_TOOLS "" CACHE BOOL OFF FORCE)
set(WITH_TOOLS "" CACHE BOOL OFF FORCE)
include(CMakePackageConfigHelpers)
add_subdirectory(${rocksdb_SOURCE_DIR} ${rocksdb_BINARY_DIR})
set_property(GLOBAL PROPERTY ROCKSDB_INCLUDE_DIR "${rocksdb_SOURCE_DIR}/include")
add_custom_target(external_rocksdb)
add_dependencies(external_rocksdb rocksdb)