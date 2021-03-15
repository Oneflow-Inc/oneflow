include (ExternalProject)

set(LIBJPEG_INCLUDE_DIR ${THIRD_PARTY_DIR}/libjpeg-turbo/include)
set(LIBJPEG_LIBRARY_DIR ${THIRD_PARTY_DIR}/libjpeg-turbo/lib)

set(LIBJPEG_URL https://github.com/Oneflow-Inc/libjpeg-turbo/archive/3041cf67f.tar.gz)
use_mirror(VARIABLE LIBJPEG_URL URL ${LIBJPEG_URL})

if(WIN32)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(LIBJPEG_BUILD_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/src/libjpeg-turbo)
    set(LIBJPEG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/src/libjpeg-turbo/${CMAKE_BUILD_TYPE})
    set(LIBJPEG_LIBRARY_NAMES libturbojpeg.a)
else()
    set(LIBJPEG_BUILD_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/src/libjpeg-turbo)
    set(LIBJPEG_BUILD_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/src/libjpeg-turbo)
    set(LIBJPEG_LIBRARY_NAMES libturbojpeg.a)
endif()

foreach(LIBRARY_NAME ${LIBJPEG_LIBRARY_NAMES})
    list(APPEND LIBJPEG_STATIC_LIBRARIES ${LIBJPEG_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND LIBJPEG_BUILD_STATIC_LIBRARIES ${LIBJPEG_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(LIBJPEG_HEADERS
     "${LIBJPEG_BUILD_SRC_DIR}/cderror.h"
     "${LIBJPEG_BUILD_SRC_DIR}/cdjpeg.h"
     "${LIBJPEG_BUILD_SRC_DIR}/cmyk.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jchuff.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jconfig.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdcoefct.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdct.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdhuff.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdmainct.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdmaster.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jdsample.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jerror.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jinclude.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jmemsys.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jmorecfg.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jpegcomp.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jpegint.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jpeglib.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jpeg_nbits_table.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jsimddct.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jsimd.h"
     "${LIBJPEG_BUILD_SRC_DIR}/jversion.h"
     "${LIBJPEG_BUILD_SRC_DIR}/tjutil.h"
     "${LIBJPEG_BUILD_SRC_DIR}/transupp.h"
     "${LIBJPEG_BUILD_SRC_DIR}/turbojpeg.h"
)

if(THIRD_PARTY)

ExternalProject_Add(libjpeg-turbo
    PREFIX libjpeg-turbo
    URL ${LIBJPEG_URL}
    URL_MD5 04e30c853d771ce0c29192579043dd86
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

# put libjpeg-turbo includes in the directory where they are expected
add_custom_target(libjpeg_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBJPEG_INCLUDE_DIR}
    DEPENDS libjpeg-turbo)

add_custom_target(libjpeg_copy_headers_to_destination
    DEPENDS libjpeg_create_header_dir)

foreach(header_file ${LIBJPEG_HEADERS})
    add_custom_command(TARGET libjpeg_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${LIBJPEG_INCLUDE_DIR})
endforeach()

# pub libjpeg libs in the directory where they are expected
add_custom_target(libjpeg_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBJPEG_LIBRARY_DIR}
    DEPENDS libjpeg-turbo)

add_custom_target(libjpeg_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBJPEG_BUILD_STATIC_LIBRARIES} ${LIBJPEG_LIBRARY_DIR}
    DEPENDS libjpeg_create_library_dir)

endif(THIRD_PARTY)
