include (ExternalProject)

set(GOOGLETEST_INCLUDE_DIR ${THIRD_PARTY_DIR}/googletest/include)
set(GOOGLETEST_LIBRARY_DIR ${THIRD_PARTY_DIR}/googletest/lib)

set(googletest_SRC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_TAG ec44c6c1675c25b9827aacd08c02433cccde7780)

if(WIN32)
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/${CMAKE_BUILD_TYPE})
    set(GOOGLETEST_LIBRARY_NAMES gtest.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/${CMAKE_BUILD_TYPE})
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a)
else()
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest)
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a)
endif()

foreach(LIBRARY_NAME ${GOOGLETEST_LIBRARY_NAMES})
    list(APPEND GOOGLETEST_STATIC_LIBRARIES ${GOOGLETEST_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GOOGLETEST_BUILD_STATIC_LIBRARIES ${GOOGLETEST_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_GMOCK:BOOL=OFF
        -DBUILD_GTEST:BOOL=ON
        #-Dgtest_force_shared_crt:BOOL=ON  #default value is OFF
)

add_custom_target(googletest_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GOOGLETEST_INCLUDE_DIR}
  DEPENDS googletest)

add_custom_target(googletest_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${googletest_SRC_INCLUDE_DIR} ${GOOGLETEST_INCLUDE_DIR}
  DEPENDS googletest_create_header_dir)

add_custom_target(googletest_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GOOGLETEST_LIBRARY_DIR}
  DEPENDS googletest)

add_custom_target(googletest_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GOOGLETEST_BUILD_STATIC_LIBRARIES} ${GOOGLETEST_LIBRARY_DIR}
  DEPENDS googletest_create_library_dir)
