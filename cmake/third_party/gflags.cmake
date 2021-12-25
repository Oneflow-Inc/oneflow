include (ExternalProject)

set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_DIR}/gflags/install)
set(GFLAGS_INCLUDE_DIR ${GFLAGS_INSTALL_DIR}/include)
set(GFLAGS_LIBRARY_DIR ${GFLAGS_INSTALL_DIR}/lib)

set(gflags_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include)
set(gflags_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/lib)
set(gflags_URL https://github.com/Oneflow-Inc/gflags/archive/9314597d4.tar.gz)
use_mirror(VARIABLE gflags_URL URL ${gflags_URL})

if(WIN32)
    set(GFLAGS_LIBRARY_NAMES gflags_static.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GFLAGS_LIBRARY_NAMES libgflags.a)
else()
    set(GFLAGS_LIBRARY_NAMES libgflags.a)
endif()

foreach(LIBRARY_NAME ${GFLAGS_LIBRARY_NAMES})
    list(APPEND GFLAGS_STATIC_LIBRARIES ${GFLAGS_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set (GFLAGS_PUBLIC_H
  ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include/gflags/config.h
  ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include/gflags/gflags_completions.h
  ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include/gflags/gflags_declare.h
  ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include/gflags/gflags.h
)

if (THIRD_PARTY)

ExternalProject_Add(gflags
    PREFIX gflags
    URL ${gflags_URL}
    URL_MD5 9677cc51d63642ba3d5f2a57a1fa2bd0
    UPDATE_COMMAND bash -c "rm -f BUILD || true"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${GFLAGS_STATIC_LIBRARIES}
    CMAKE_CACHE_ARGS
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
        -DGFLAGS_NAMESPACE:STRING=gflags
        -DCMAKE_INSTALL_PREFIX:STRING=${GFLAGS_INSTALL_DIR}
        -DCMAKE_INSTALL_MESSAGE:STRING=${CMAKE_INSTALL_MESSAGE}
)

endif(THIRD_PARTY)
add_library(gflags_imported UNKNOWN IMPORTED)
set_property(TARGET gflags_imported PROPERTY IMPORTED_LOCATION "${GFLAGS_STATIC_LIBRARIES}")
