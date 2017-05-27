include (ExternalProject)

set(TENSORFLOW_INCLUDE_DIR ${THIRD_PARTY_DIR}/tensorflow/include)
set(TENSORFLOW_LIBRARY_DIR ${THIRD_PARTY_DIR}/tensorflow/lib)

set(TENSORFLOW_URL https://github.com/Oneflow-Inc/tensorflow.git)
set(TENSORFLOW_TAG 71d873dd3514220b8ef2c3608d292aeeb50ec3a5)

if(WIN32)
    set(TENSORFLOW_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/${CMAKE_BUILD_TYPE})
    set(TENSORFLOW_LIBRARY_NAME tensorflow.lib)
    set(TENSORFLOW_ADDITIONAL_CMAKE_OPTIONS -A x64)

elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))

else()

endif()

if(BUILD_THIRD_PARTY)

add_custom_target(tensorflow_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_INCLUDE_DIR})

add_custom_target(tensorflow_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_LIBRARY_DIR})

ExternalProject_Add(tensorflow
    PREFIX tensorflow
    DEPENDS zlib_copy_headers_to_destination
            zlib_copy_libs_to_destination
            protobuf_copy_headers_to_destination
            protobuf_copy_libs_to_destination
            protobuf_copy_binary_to_destination
            tensorflow_create_header_dir
            tensorflow_create_library_dir
    GIT_REPOSITORY ${TENSORFLOW_URL}
    GIT_TAG ${TENSORFLOW_TAG}
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} tensorflow/contrib/cmake/
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_VERBOSE_MAKEFILE=OFF
        -DZLIB_INSTALL=${ZLIB_INSTALL}
        -Dzlib_INCLUDE_DIR=${ZLIB_INCLUDE_DIR}
        -Dzlib_STATIC_LIBRARIES=${ZLIB_STATIC_LIBRARIES}
        -DPROTOBUF_INCLUDE_DIRS=${PROTOBUF_INCLUDE_DIR}
        -Dprotobuf_STATIC_LIBRARIES=${PROTOBUF_STATIC_LIBRARIES}
        -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOBUF_PROTOC_EXECUTABLE}
        ${TENSORFLOW_ADDITIONAL_CMAKE_OPTIONS}        
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
        -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DZLIB_INSTALL:STRING=${ZLIB_INSTALL}
        -Dzlib_INCLUDE_DIR:STRING=${ZLIB_INCLUDE_DIR}
        -Dzlib_STATIC_LIBRARIES:STRING=${ZLIB_STATIC_LIBRARIES}
        -DPROTOBUF_INCLUDE_DIRS:STRING=${PROTOBUF_INCLUDE_DIR}
        -Dprotobuf_STATIC_LIBRARIES:STRING=${PROTOBUF_STATIC_LIBRARIES}
        -DPROTOBUF_PROTOC_EXECUTABLE:STRING=${PROTOBUF_PROTOC_EXECUTABLE}
        )



endif(BUILD_THIRD_PARTY)