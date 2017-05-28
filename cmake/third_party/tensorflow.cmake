include (ExternalProject)

option(tensorflow_BUILD_CC_TESTS "Build cc unit tests " ON)
option(tensorflow_ENABLE_GPU "Enable GPU support" OFF)

set(TENSORFLOW_URL https://github.com/Oneflow-Inc/tensorflow.git)
set(TENSORFLOW_TAG 71d873dd3514220b8ef2c3608d292aeeb50ec3a5)

set(TENSORFLOW_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow)
set(TENSORFLOW_INCLUDE_DIR ${THIRD_PARTY_DIR}/tensorflow/include)
set(TENSORFLOW_LIBRARY_DIR ${THIRD_PARTY_DIR}/tensorflow/lib)
set(tensorflow_source_dir ${THIRD_PARTY_DIR}/tensorflow/include)

set(GIF_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/gif/install/include)
set(GIF_INCLUDE_DIR ${THIRD_PARTY_DIR}/gif/include)
set(GIF_LIBRARY_DIR ${THIRD_PARTY_DIR}/gif/lib)
set(GIF_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/gif/install/lib)

set(FARMHASH_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/farmhash/install/include)
set(FARMHASH_INCLUDE_DIR ${THIRD_PARTY_DIR}/farmhash/include)
set(FARMHASH_LIBRARY_DIR ${THIRD_PARTY_DIR}/farmhash/lib)
set(FARMHASH_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/farmhash/install/lib)

set(HIGHWAYHASH_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/highwayhash/install/include)
set(HIGHWAYHASH_INCLUDE_DIR ${THIRD_PARTY_DIR}/highwayhash/include)
set(HIGHWAYHASH_LIBRARY_DIR ${THIRD_PARTY_DIR}/highwayhash/lib)
set(HIGHWAYHASH_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/highwayhash/install/lib)

set(JPEG_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jpeg/install/include)
set(JPEG_INCLUDE_DIR ${THIRD_PARTY_DIR}/jpeg/include)
set(JPEG_LIBRARY_DIR ${THIRD_PARTY_DIR}/jpeg/lib)
set(JPEG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jpeg/install/lib)

set(PNG_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/png/install/include)
set(PNG_INCLUDE_DIR ${THIRD_PARTY_DIR}/png/include)
set(PNG_LIBRARY_DIR ${THIRD_PARTY_DIR}/png/lib)
set(PNG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/png/install/lib)

set(JSONCPP_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jsoncpp/src/jsoncpp/include/json)
set(JSONCPP_INCLUDE_DIR ${THIRD_PARTY_DIR}/jsoncpp/include)
set(JSONCPP_LIBRARY_DIR ${THIRD_PARTY_DIR}/jsoncpp/lib)

set(THIRD_PARTY_EIGEN3_DIR ${THIRD_PARTY_DIR}/eigen3)
set(THIRD_PARTY_EIGEN3_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/third_party/eigen3)

set(EIGEN_ARCHIVE_DIR ${THIRD_PARTY_DIR}/eigen_archive)
set(EIGEN_ARCHIVE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/external/eigen_archive)

set(EIGEN_INCLUDE_DIRS
    #${THIRD_PARTY_EIGEN3_DIR}
    ${EIGEN_ARCHIVE_DIR})

if(WIN32)
    set(TENSORFLOW_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/${CMAKE_BUILD_TYPE})
    set(TENSORFLOW_LIBRARY_NAME "tensorflow.lib")


    set(GIF_LIBRARY_NAME "giflib.lib")
    set(FARMHASH_LIBRARY_NAME "farmhash.lib")
    set(HIGHWAYHASH_LIBRARY_NAME "highwayhash.lib")
    set(JPEG_LIBRARY_NAME "libjpeg.lib")
    set(PNG_LIBRARY_NAME "libpng12_staticd.lib")

    set(JSONCPP_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jsoncpp/src/jsoncpp/src/lib_json/${CMAKE_BUILD_TYPE})
    set(JSONCPP_LIBRARY_NAME "jsoncpp.lib")

    set(TENSORFLOW_ADDITIONAL_CMAKE_OPTIONS -A x64)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(TENSORFLOW_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/${CMAKE_BUILD_TYPE})
    set(TENSORFLOW_LIBRARY_NAME "libtensorflow.a")


    set(GIF_LIBRARY_NAME "libgif.a")
    set(FARMHASH_LIBRARY_NAME "libfarmhash.a")
    set(HIGHWAYHASH_LIBRARY_NAME "libhighwayhash.a")
    set(JPEG_LIBRARY_NAME "libjpeg.a")
    set(PNG_LIBRARY_NAME "libpng12.a")

    set(JSONCPP_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jsoncpp/src/jsoncpp/src/lib_json/${CMAKE_BUILD_TYPE})
    set(JSONCPP_LIBRARY_NAME "libjsoncpp.a")

else()
    set(TENSORFLOW_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow)
    set(TENSORFLOW_LIBRARY_NAME "libtensorflow.a")

    set(GIF_LIBRARY_NAME "libgif.a")
    set(FARMHASH_LIBRARY_NAME "libfarmhash.a")
    set(HIGHWAYHASH_LIBRARY_NAME "libhighwayhash.a")
    set(JPEG_LIBRARY_NAME "libjpeg.a")
    set(PNG_LIBRARY_NAME "libpng12.a")

    set(JSONCPP_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/src/tensorflow/jsoncpp/src/jsoncpp/src/lib_json)
    set(JSONCPP_LIBRARY_NAME "libjsoncpp.a")
endif()

set(TENSORFLOW_BUILD_LIBRARY ${TENSORFLOW_BUILD_LIBRARY_DIR}/${TENSORFLOW_LIBRARY_NAME})
set(tensorflow_STATIC_LIBRARIES ${TENSORFLOW_LIBRARY_DIR}/${TENSORFLOW_LIBRARY_NAME})

set(GIF_BUILD_LIBRARY ${GIF_BUILD_LIBRARY_DIR}/${GIF_LIBRARY_NAME})
set(gif_STATIC_LIBRARIES ${GIF_LIBRARY_DIR}/${GIF_LIBRARY_NAME})

set(FARMHASH_BUILD_LIBRARY ${FARMHASH_BUILD_LIBRARY_DIR}/${FARMHASH_LIBRARY_NAME})
set(farmhash_STATIC_LIBRARIES ${FARMHASH_LIBRARY_DIR}/${FARMHASH_LIBRARY_NAME})

set(HIGHWAYHASH_BUILD_LIBRARY ${HIGHWAYHASH_BUILD_LIBRARY_DIR}/${HIGHWAYHASH_LIBRARY_NAME})
set(highwayhash_STATIC_LIBRARIES ${HIGHWAYHASH_LIBRARY_DIR}/${HIGHWAYHASH_LIBRARY_NAME})

set(JPEG_BUILD_LIBRARY ${JPEG_BUILD_LIBRARY_DIR}/${JPEG_LIBRARY_NAME})
set(JPEG_STATIC_LIBRARIES ${JPEG_LIBRARY_DIR}/${JPEG_LIBRARY_NAME})

set(PNG_BUILD_LIBRARY ${PNG_BUILD_LIBRARY_DIR}/${PNG_LIBRARY_NAME})
set(PNG_STATIC_LIBRARIES ${PNG_LIBRARY_DIR}/${PNG_LIBRARY_NAME})

set(JSONCPP_BUILD_LIBRARY ${JSONCPP_BUILD_LIBRARY_DIR}/${JSONCPP_LIBRARY_NAME})
set(JSONCPP_STATIC_LIBRARIES ${JSONCPP_LIBRARY_DIR}/${JSONCPP_LIBRARY_NAME})


if(BUILD_THIRD_PARTY)

ExternalProject_Add(tensorflow
    PREFIX tensorflow
    DEPENDS zlib_copy_headers_to_destination
            zlib_copy_libs_to_destination
            protobuf_copy_headers_to_destination
            protobuf_copy_libs_to_destination
            protobuf_copy_binary_to_destination
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

add_custom_target(tensorflow_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_INCLUDE_DIR}/tensorflow
    DEPENDS tensorflow)
add_custom_target(tensorflow_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TENSORFLOW_BUILD_INCLUDE} ${TENSORFLOW_INCLUDE_DIR}/tensorflow
    DEPENDS tensorflow_create_header_dir)


add_custom_target(tensorflow_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TENSORFLOW_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(tensorflow_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORFLOW_BUILD_LIBRARY} ${TENSORFLOW_LIBRARY_DIR}
    DEPENDS tensorflow_create_library_dir)

add_custom_target(gif_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GIF_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(gif_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${GIF_BUILD_INCLUDE} ${GIF_INCLUDE_DIR}
    DEPENDS gif_create_header_dir)


add_custom_target(gif_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GIF_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(gif_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GIF_BUILD_LIBRARY} ${GIF_LIBRARY_DIR}
    DEPENDS gif_create_library_dir)

add_custom_target(farmhash_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FARMHASH_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(farmhash_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${FARMHASH_BUILD_INCLUDE} ${FARMHASH_INCLUDE_DIR}
    DEPENDS farmhash_create_header_dir)


add_custom_target(farmhash_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FARMHASH_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(farmhash_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FARMHASH_BUILD_LIBRARY} ${FARMHASH_LIBRARY_DIR}
    DEPENDS farmhash_create_library_dir)


add_custom_target(highwayhash_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HIGHWAYHASH_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(highwayhash_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${HIGHWAYHASH_BUILD_INCLUDE} ${HIGHWAYHASH_INCLUDE_DIR}
    DEPENDS highwayhash_create_header_dir)


add_custom_target(highwayhash_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HIGHWAYHASH_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(highwayhash_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HIGHWAYHASH_BUILD_LIBRARY} ${HIGHWAYHASH_LIBRARY_DIR}
    DEPENDS highwayhash_create_library_dir)

add_custom_target(jpeg_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JPEG_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(jpeg_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${JPEG_BUILD_INCLUDE} ${JPEG_INCLUDE_DIR}
    DEPENDS jpeg_create_header_dir)


add_custom_target(jpeg_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JPEG_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(jpeg_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${JPEG_BUILD_LIBRARY} ${JPEG_LIBRARY_DIR}
    DEPENDS jpeg_create_library_dir)

add_custom_target(png_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PNG_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(png_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PNG_BUILD_INCLUDE} ${PNG_INCLUDE_DIR}
    DEPENDS png_create_header_dir)


add_custom_target(png_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PNG_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(png_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PNG_BUILD_LIBRARY} ${PNG_LIBRARY_DIR}
    DEPENDS png_create_library_dir)

add_custom_target(jsoncpp_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JSONCPP_INCLUDE_DIR}
    DEPENDS tensorflow)
add_custom_target(jsoncpp_copy_headers_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${JSONCPP_BUILD_INCLUDE} ${JSONCPP_INCLUDE_DIR}
    DEPENDS jsoncpp_create_header_dir)


add_custom_target(jsoncpp_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JSONCPP_LIBRARY_DIR}
    DEPENDS tensorflow)
add_custom_target(jsoncpp_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${JSONCPP_BUILD_LIBRARY} ${JSONCPP_LIBRARY_DIR}
    DEPENDS jsoncpp_create_library_dir)

add_custom_target(eigen_create_headers_dir
    DEPENDS tensorflow)
add_custom_command(TARGET eigen_create_headers_dir PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${THIRD_PARTY_EIGEN3_DIR})
add_custom_command(TARGET eigen_create_headers_dir PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EIGEN_ARCHIVE_DIR})

add_custom_target(eigen_copy_headers_dir
    DEPENDS eigen_create_headers_dir)
add_custom_command(TARGET eigen_copy_headers_dir PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${THIRD_PARTY_EIGEN3_BUILD_DIR} ${THIRD_PARTY_EIGEN3_DIR})
add_custom_command(TARGET eigen_copy_headers_dir PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${EIGEN_ARCHIVE_BUILD_DIR} ${EIGEN_ARCHIVE_DIR})

endif(BUILD_THIRD_PARTY)