include (ExternalProject)

set(GFLAGS_INCLUDE_DIR ${THIRD_PARTY_DIR}/gflags/include)
set(GFLAGS_LIBRARY_DIR ${THIRD_PARTY_DIR}/gflags/lib)

set(gflags_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/include)
set(gflags_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/gflags/src/gflags/lib)
set(gflags_URL ${THIRD_PARTY_SUBMODULE_DIR}/gflags/src/gflags)

if(WIN32)
    set(GFLAGS_BUILD_LIBRARY_DIR ${gflags_LIB_DIR}/${CMAKE_BUILD_TYPE})
    set(GFLAGS_LIBRARY_NAMES gflags_static.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GFLAGS_BUILD_LIBRARY_DIR ${gflags_LIB_DIR}/${CMAKE_BUILD_TYPE})
    set(GFLAGS_LIBRARY_NAMES libgflags.a)
else()
    set(GFLAGS_BUILD_LIBRARY_DIR ${gflags_LIB_DIR})
    set(GFLAGS_LIBRARY_NAMES libgflags.a)
endif()

foreach(LIBRARY_NAME ${GFLAGS_LIBRARY_NAMES})
    list(APPEND GFLAGS_STATIC_LIBRARIES ${GFLAGS_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GFLAGS_BUILD_STATIC_LIBRARIES ${GFLAGS_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if (THIRD_PARTY)

# TODO: investigate if these three lines are necessary
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

ExternalProject_Add(gflags
    PREFIX gflags
    URL ${gflags_URL}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS:STRING=-fPIC
        -DGFLAGS_NAMESPACE:STRING=gflags
)


add_custom_target(gflags_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GFLAGS_INCLUDE_DIR}
  DEPENDS gflags)

add_custom_target(gflags_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${gflags_HEADERS_DIR} ${GFLAGS_INCLUDE_DIR}
    DEPENDS gflags_create_header_dir)

add_custom_target(gflags_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GFLAGS_LIBRARY_DIR}
  DEPENDS gflags)

add_custom_target(gflags_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GFLAGS_BUILD_STATIC_LIBRARIES} ${GFLAGS_LIBRARY_DIR}
  DEPENDS gflags_create_library_dir)

endif(THIRD_PARTY)
