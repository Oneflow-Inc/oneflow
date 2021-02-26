include (ExternalProject)

set(NCCL_INCLUDE_DIR ${THIRD_PARTY_DIR}/nccl/include)
set(NCCL_LIBRARY_DIR ${THIRD_PARTY_DIR}/nccl/lib)

set(NCCL_URL https://github.com/NVIDIA/nccl/archive/v2.8.3-1.tar.gz)
use_mirror(VARIABLE NCCL_URL URL ${NCCL_URL})
set(NCCL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/nccl/src/nccl/build)

if(WIN32)
    set(NCCL_BUILD_LIBRARY_DIR ${NCCL_BUILD_DIR}/lib)
    set(NCCL_LIBRARY_NAMES libnccl_static.lib)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
    set(NCCL_BUILD_LIBRARY_DIR ${NCCL_BUILD_DIR}/lib)
    set(NCCL_LIBRARY_NAMES libnccl_static.a)
else()
    set(NCCL_BUILD_LIBRARY_DIR ${NCCL_BUILD_DIR}/lib)
    set(NCCL_LIBRARY_NAMES libnccl_static.a)
endif()

foreach(LIBRARY_NAME ${NCCL_LIBRARY_NAMES})
    list(APPEND NCCL_STATIC_LIBRARIES ${NCCL_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND NCCL_BUILD_STATIC_LIBRARIES ${NCCL_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(NCCL_HEADERS
    "${NCCL_BUILD_DIR}/include/nccl.h"
)

if(THIRD_PARTY)

include(ProcessorCount)
ProcessorCount(PROC_NUM)
ExternalProject_Add(nccl
    PREFIX nccl
    URL ${NCCL_URL}
    URL_MD5 0dfa6079a4bded866fa0f94281c58581
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${PROC_NUM} src.build CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
    INSTALL_COMMAND ""
)

# put nccl includes in the directory where they are expected
add_custom_target(nccl_create_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NCCL_INCLUDE_DIR}
    DEPENDS nccl)

add_custom_target(nccl_copy_headers_to_destination
    DEPENDS nccl_create_header_dir)

foreach(header_file ${NCCL_HEADERS})
    add_custom_command(TARGET nccl_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${NCCL_INCLUDE_DIR})
endforeach()

# pub nccl libs in the directory where they are expected
add_custom_target(nccl_create_library_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NCCL_LIBRARY_DIR}
    DEPENDS nccl)

add_custom_target(nccl_copy_libs_to_destination
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NCCL_BUILD_STATIC_LIBRARIES} ${NCCL_LIBRARY_DIR}
    DEPENDS nccl_create_library_dir)

endif(THIRD_PARTY)
