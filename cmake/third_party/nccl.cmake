include (ExternalProject)

option(NCCL_STATIC "" ON)
if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
   set(NCCL_STATIC OFF)
endif()

set(NCCL_INSTALL_DIR ${THIRD_PARTY_DIR}/nccl)
set(NCCL_INCLUDE_DIR ${NCCL_INSTALL_DIR}/include)
set(NCCL_LIBRARY_DIR ${NCCL_INSTALL_DIR}/lib)

set(NCCL_URL https://github.com/NVIDIA/nccl/archive/refs/tags/v2.9.8-1.tar.gz)
use_mirror(VARIABLE NCCL_URL URL ${NCCL_URL})

if(WIN32)
    set(NCCL_LIBRARY_NAMES libnccl_static.lib)
else()
    if(NCCL_STATIC)
        set(NCCL_LIBRARY_NAMES libnccl_static.a)
    else()
        set(NCCL_LIBRARY_NAMES libnccl.so)
    endif()
endif()

foreach(LIBRARY_NAME ${NCCL_LIBRARY_NAMES})
    list(APPEND NCCL_LIBRARIES ${NCCL_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

if(THIRD_PARTY)

include(ProcessorCount)
ProcessorCount(PROC_NUM)
ExternalProject_Add(nccl
    PREFIX nccl
    URL ${NCCL_URL}
    URL_MD5 9894dffc51d9d276f01286094ac220ac
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${PROC_NUM} src.build CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
    INSTALL_COMMAND make src.install PREFIX=${NCCL_INSTALL_DIR}
)

endif(THIRD_PARTY)
