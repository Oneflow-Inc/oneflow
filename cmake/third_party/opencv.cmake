include (ExternalProject)

set(OPENCV_INCLUDE_DIR ${THIRD_PARTY_DIR}/opencv/include)
set(LIBPNG_INCLUDE_DIR ${THIRD_PARTY_DIR}/libpng/include)
set(OPENCV_LIBRARY_DIR ${THIRD_PARTY_DIR}/opencv/lib)
set(OPENCV_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv/build/install)

set(OPENCV_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv/src)
set(OPENCV_URL https://github.com/Oneflow-Inc/opencv/archive/51cef2651.tar.gz)
use_mirror(VARIABLE OPENCV_URL URL ${OPENCV_URL})

if(WIN32)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
else()
    include(GNUInstallDirs)
    set(OPENCV_BUILD_INCLUDE_DIR ${OPENCV_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
    set(OPENCV_BUILD_LIBRARY_DIR ${OPENCV_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(OPENCV_BUILD_3RDPARTY_LIBRARY_DIR ${OPENCV_INSTALL_DIR}/share/OpenCV/3rdparty/${CMAKE_INSTALL_LIBDIR})
    set(OPENCV_LIBRARY_NAMES libopencv_imgproc.a libopencv_highgui.a libopencv_imgcodecs.a libopencv_core.a)
    set(OPENCV_3RDPARTY_LIBRARY_NAMES libIlmImf.a liblibjasper.a liblibpng.a liblibtiff.a liblibwebp.a)
endif()

foreach(LIBRARY_NAME ${OPENCV_LIBRARY_NAMES})
    list(APPEND OPENCV_STATIC_LIBRARIES ${OPENCV_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND OPENCV_BUILD_STATIC_LIBRARIES ${OPENCV_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

foreach(LIBRARY_NAME ${OPENCV_3RDPARTY_LIBRARY_NAMES})
    list(APPEND OPENCV_STATIC_LIBRARIES ${OPENCV_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND OPENCV_BUILD_STATIC_LIBRARIES ${OPENCV_BUILD_3RDPARTY_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()


if (THIRD_PARTY)

ExternalProject_Add(opencv
    DEPENDS libjpeg_copy_headers_to_destination libjpeg_copy_libs_to_destination
    PREFIX opencv
    URL ${OPENCV_URL}
    URL_MD5 59870e55385f5202c1aa178fe37ed2de
    UPDATE_COMMAND ""
    PATCH_COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv/build
    BUILD_IN_SOURCE 0
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv/build
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${OPENCV_INSTALL_DIR}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DWITH_IPP:BOOL=OFF
        -DWITH_1394:BOOL=OFF
        -DWITH_AVFOUNDATION:BOOL=OFF
        -DWITH_CAROTENE:BOOL=OFF
        -DWITH_CPUFEATURES:BOOL=OFF
        -DWITH_VTK:BOOL=OFF
        -DWITH_CUDA:BOOL=OFF
        -DWITH_CUFFT:BOOL=OFF
        -DWITH_CUBLAS:BOOL=OFF
        -DWITH_NVCUVID:BOOL=OFF
        -DWITH_EIGEN:BOOL=OFF
        -DWITH_VFW:BOOL=OFF
        -DWITH_FFMPEG:BOOL=OFF
        -DWITH_WEBP:BOOL=ON
        -DBUILD_WEBP:BOOL=ON
        -DWITH_GSTREAMER:BOOL=OFF
        -DWITH_GSTREAMER_0_10:BOOL=OFF
        -DWITH_GTK:BOOL=OFF
        -DWITH_GTK_2_X:BOOL=OFF
        -DWITH_WIN32UI:BOOL=OFF
        -DWITH_PTHREADS_PF:BOOL=OFF
        -DWITH_DSHOW:BOOL=OFF
        -DWITH_OPENCL:BOOL=OFF
        -DWITH_OPENCL_SVM:BOOL=OFF
        -DWITH_OPENCLAMDFFT:BOOL=OFF
        -DWITH_OPENCLAMDBLAS:BOOL=OFF
        -DWITH_DIRECTX:BOOL=OFF
        -DWITH_MATLAB:BOOL=OFF
        -DWITH_GPHOTO2:BOOL=OFF
        -DWITH_LAPACK:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DBUILD_ANDROID_EXAMPLES:BOOL=OFF
        -DBUILD_DOCS:BOOL=OFF
        -DBUILD_PACKAGE:BOOL=OFF
        -DBUILD_PERF_TESTS:BOOL=OFF
        -DBUILD_TESTS:BOOL=OFF
        -DBUILD_FAT_JAVA_LIBS:BOOL=OFF
        -DBUILD_ANDROID_SERVICE:BOOL=OFF
        -DBUILD_CUDA_STUBS:BOOL=OFF
        -DENABLE_PYLINT:BOOL=OFF
        -DBUILD_opencv_python3:BOOL=OFF
        -DBUILD_opencv_python2:BOOL=OFF
        -DBUILD_opencv_world:BOOL=OFF
        -DBUILD_opencv_apps:BOOL=OFF
        -DBUILD_opencv_js:BOOL=OFF
        -DBUILD_ZLIB:BOOL=ON
        -DBUILD_TIFF:BOOL=ON
        -DBUILD_JASPER:BOOL=ON
        -DWITH_JPEG:BOOL=ON
        -DBUILD_JPEG:BOOL=OFF
        -DJPEG_INCLUDE_DIR:STRING=${LIBJPEG_INCLUDE_DIR}
        -DJPEG_LIBRARY:STRING=${LIBJPEG_STATIC_LIBRARIES}
        -DBUILD_PNG:BOOL=ON
        -DBUILD_OPENEXR:BOOL=ON
        -DBUILD_TBB:BOOL=ON
        -DBUILD_IPP_IW:BOOL=OFF
        -DWITH_ITT:BOOL=ON
        # -DLIB_SUFFIX:STRING=64
)

# put opencv includes in the 'THIRD_PARTY_DIR'
add_copy_headers_target(NAME opencv SRC ${OPENCV_BUILD_INCLUDE_DIR} DST ${OPENCV_INCLUDE_DIR} DEPS opencv INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/opencv_headers.txt")

add_copy_headers_target(NAME libpng SRC ${CMAKE_CURRENT_BINARY_DIR}/opencv/src/opencv/3rdparty/libpng DST ${LIBPNG_INCLUDE_DIR} DEPS opencv INDEX_FILE "${oneflow_cmake_dir}/third_party/header_index/libpng_headers.txt")

# put opencv librarys in the 'THIRD_PARTY_DIR'
add_custom_target(opencv_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENCV_LIBRARY_DIR}
  DEPENDS opencv)

add_custom_target(opencv_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_BUILD_STATIC_LIBRARIES} ${OPENCV_LIBRARY_DIR}
  DEPENDS opencv_create_library_dir)

endif(THIRD_PARTY)
