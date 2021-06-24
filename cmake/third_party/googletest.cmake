include (ExternalProject)

set(GOOGLETEST_INSTALL_DIR ${THIRD_PARTY_DIR}/gtest)

set(googletest_SRC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googlemock_SRC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/include)

set(googletest_URL https://github.com/google/googletest/archive/release-1.8.0.tar.gz)
use_mirror(VARIABLE googletest_URL URL ${googletest_URL})

if(WIN32)
    set(GOOGLETEST_LIBRARY_NAMES gtest.lib gtest_main.lib)
    set(GOOGLEMOCK_LIBRARY_NAMES gmock.lib gmock_main.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a libgtest_main.a)
    set(GOOGLEMOCK_LIBRARY_NAMES libgmock.a libgmock_main.a)
else()
    set(GOOGLETEST_LIBRARY_NAMES libgtest.a libgtest_main.a)
    set(GOOGLEMOCK_LIBRARY_NAMES libgmock.a libgmock_main.a)
endif()

foreach(LIBRARY_NAME ${GOOGLETEST_LIBRARY_NAMES})
    list(APPEND GOOGLETEST_STATIC_LIBRARIES ${GOOGLETEST_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

foreach(LIBRARY_NAME ${GOOGLEMOCK_LIBRARY_NAMES})
    list(APPEND GOOGLEMOCK_STATIC_LIBRARIES ${GOOGLEMOCK_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)

ExternalProject_Add(googletest
    PREFIX googletest
    URL ${googletest_URL}
    URL_MD5 16877098823401d1bf2ed7891d7dce36
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
        -DBUILD_GMOCK:BOOL=ON
	      -DBUILD_GTEST:BOOL=OFF  # gmock includes gtest
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_INSTALL_PREFIX:STRING=${GOOGLETEST_INSTALL_DIR}
        #-Dgtest_force_shared_crt:BOOL=ON  #default value is OFF
)

endif(THIRD_PARTY)
