include (ExternalProject)

set(GOOGLETEST_INCLUDE_DIR ${THIRD_PARTY_DIR}/googletest/include)
set(GOOGLETEST_LIBRARY_DIR ${THIRD_PARTY_DIR}/googletest/lib)
set(GOOGLEMOCK_INCLUDE_DIR ${THIRD_PARTY_DIR}/googlemock/include)
set(GOOGLEMOCK_LIBRARY_DIR ${THIRD_PARTY_DIR}/googlemock/lib)


set(googletest_SRC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googlemock_SRC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/include)

set(googletest_URL https://github.com/google/googletest.git)
set(googletest_TAG ec44c6c1675c25b9827aacd08c02433cccde7780)

if(WIN32)
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/${CMAKE_BUILD_TYPE})
    set(GOOGLETEST_LIBRARY_NAMES gtest.lib gtest_main.lib)
    set(GOOGLEMOCK_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/${CMAKE_BUILD_TYPE})
    set(GOOGLEMOCK_LIBRARY_NAMES gmock.lib gmock_main.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/${CMAKE_BUILD_TYPE})
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a libgtest_main.a)
    set(GOOGLEMOCK_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/${CMAKE_BUILD_TYPE})
    set(GOOGLEMOCK_LIBRARY_NAMES libgmock.a libgmock_main.a)
else()
    set(GOOGLETEST_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest)
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a libgtest_main.a)
    set(GOOGLEMOCK_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock)
    set(GOOGLEMOCK_LIBRARY_NAMES libgmock.a libgmock_main.a)
endif()

foreach(LIBRARY_NAME ${GOOGLETEST_LIBRARY_NAMES})
    list(APPEND GOOGLETEST_STATIC_LIBRARIES ${GOOGLETEST_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GOOGLETEST_BUILD_STATIC_LIBRARIES ${GOOGLETEST_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

foreach(LIBRARY_NAME ${GOOGLEMOCK_LIBRARY_NAMES})
    list(APPEND GOOGLEMOCK_STATIC_LIBRARIES ${GOOGLEMOCK_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND GOOGLEMOCK_BUILD_STATIC_LIBRARIES ${GOOGLEMOCK_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(BUILD_THIRD_PARTY)

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_GMOCK:BOOL=ON
        -DBUILD_GTEST:BOOL=ON
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
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

add_custom_target(googlemock_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GOOGLEMOCK_INCLUDE_DIR}
  DEPENDS googletest)

add_custom_target(googlemock_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${googlemock_SRC_INCLUDE_DIR} ${GOOGLEMOCK_INCLUDE_DIR}
  DEPENDS googlemock_create_header_dir)

add_custom_target(googlemock_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GOOGLEMOCK_LIBRARY_DIR}
  DEPENDS googletest)

add_custom_target(googlemock_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GOOGLEMOCK_BUILD_STATIC_LIBRARIES} ${GOOGLEMOCK_LIBRARY_DIR}
  DEPENDS googlemock_create_library_dir)

endif(BUILD_THIRD_PARTY)
