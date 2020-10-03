include (ExternalProject)

set(GLOG_INCLUDE_DIR ${THIRD_PARTY_DIR}/glog/include)
set(GLOG_LIBRARY_DIR ${THIRD_PARTY_DIR}/glog/lib)

set(glog_URL https://github.com/Oneflow-Inc/glog/archive/4f3e18bf2.tar.gz)
use_mirror(VARIABLE glog_URL URL ${glog_URL})

if(WIN32)
    set(GLOG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/${CMAKE_BUILD_TYPE})
    set(GLOG_LIBRARY_NAMES glog.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GLOG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/${CMAKE_BUILD_TYPE})
    set(GLOG_LIBRARY_NAMES libglog.a)
else()
    set(GLOG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog)
    set(GLOG_LIBRARY_NAMES libglog.a)
endif()

foreach(LIBRARY_NAME ${GLOG_LIBRARY_NAMES})
    list(APPEND GLOG_STATIC_LIBRARIES ${GLOG_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GLOG_BUILD_STATIC_LIBRARIES ${GLOG_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set (GLOG_PUBLIC_H
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/config.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/raw_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/stl_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/glog/vlog_is_on.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/src/glog/src/glog/log_severity.h
)

if(THIRD_PARTY)

ExternalProject_Add(glog
    DEPENDS gflags_copy_headers_to_destination gflags_copy_libs_to_destination
    PREFIX glog
    URL ${glog_URL}
    URL_MD5 3ca928ef755c0a890680e023e3d4b9a6
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DBUILD_TESTING:BOOL=OFF
        -DWITH_GFLAGS:BOOL=ON
        -Dgflags_DIR:STRING=${oneflow_cmake_dir}/third_party
        -DMY_GFLAGS_INCLUDE_DIR:STRING=${GFLAGS_INCLUDE_DIR}
        -DMY_GFLAGS_LIBS:STRING=${GFLAGS_STATIC_LIBRARIES}
)

add_custom_target(glog_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GLOG_INCLUDE_DIR}/glog
  DEPENDS glog)

add_custom_target(glog_copy_headers_to_destination
    DEPENDS glog_create_header_dir)

foreach(header_file ${GLOG_PUBLIC_H})
    add_custom_command(TARGET glog_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${GLOG_INCLUDE_DIR}/glog)
endforeach()

add_custom_target(glog_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GLOG_LIBRARY_DIR}
  DEPENDS glog)

add_custom_target(glog_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GLOG_BUILD_STATIC_LIBRARIES} ${GLOG_LIBRARY_DIR}
  DEPENDS glog_create_library_dir)

endif(THIRD_PARTY)
