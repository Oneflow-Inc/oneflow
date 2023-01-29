option(NCCL_STATIC "" ON)
if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
  set(NCCL_STATIC OFF)
endif()
option(USE_SYSTEM_NCCL "" OFF)
set(NCCL_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NCCL")

if(WIN32)
  set(NCCL_LIBRARY_NAME libnccl_static.lib)
else()
  if(NCCL_STATIC)
    set(NCCL_LIBRARY_NAME libnccl_static.a)
  else()
    set(NCCL_LIBRARY_NAME libnccl.so)
  endif()
endif()

if(USE_SYSTEM_NCCL)
  include(FindPackageHandleStandardArgs)
  find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS}
            PATH_SUFFIXES cuda/include include)
  unset(NCCL_LIBRARY CACHE)
  find_library(
    NCCL_LIBRARY ${NCCL_LIBRARY_NAME} HINTS ${NCCL_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR}
                                            ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
  find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  add_custom_target(nccl)
else()
  get_filename_component(CUDATOOLKIT_BIN_ROOT ${CUDAToolkit_BIN_DIR} DIRECTORY)
  include(ExternalProject)
  set(NCCL_INSTALL_DIR ${THIRD_PARTY_DIR}/nccl)
  set(NCCL_INCLUDE_DIR ${NCCL_INSTALL_DIR}/include)
  set(NCCL_LIBRARY_DIR ${NCCL_INSTALL_DIR}/lib)

  # Versions 2.13 and above may cause deadlocks
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
    set(NCCL_URL https://github.com/NVIDIA/nccl/archive/refs/tags/v2.15.1-1.tar.gz)
    set(NCCL_MD5 37b787ff8934cd9374b4612f663c17fa)
  else()
    set(NCCL_URL https://github.com/NVIDIA/nccl/archive/refs/tags/v2.12.10-1.tar.gz)
    set(NCCL_MD5 bdb91f80b78c99831f09ca8bb28a1032)
  endif()

  use_mirror(VARIABLE NCCL_URL URL ${NCCL_URL})

  list(APPEND NCCL_LIBRARIES ${NCCL_LIBRARY_DIR}/${NCCL_LIBRARY_NAME})

  set(NCCL_ARCHS_LIST ${CUDA_REAL_ARCHS_LIST})

  # remove redundant archs, https://github.com/NVIDIA/nccl/blob/cb111f764a6d46370f24f75101d6b219bb2dda54/makefiles/common.mk#L28
  if("70" IN_LIST NCCL_ARCHS_LIST AND "75" IN_LIST NCCL_ARCHS_LIST)
    list(REMOVE_ITEM NCCL_ARCHS_LIST "75")
  endif()
  if("80" IN_LIST NCCL_ARCHS_LIST AND "86" IN_LIST NCCL_ARCHS_LIST)
    list(REMOVE_ITEM NCCL_ARCHS_LIST "86")
  endif()
  if("80" IN_LIST NCCL_ARCHS_LIST AND "89" IN_LIST NCCL_ARCHS_LIST)
    list(REMOVE_ITEM NCCL_ARCHS_LIST "89")
  endif()

  foreach(arch ${NCCL_ARCHS_LIST})
    string(APPEND NCCL_GENCODE "-gencode=arch=compute_${arch},code=sm_${arch} ")
  endforeach()

  if(THIRD_PARTY)

    include(ProcessorCount)
    ProcessorCount(PROC_NUM)
    ExternalProject_Add(
      nccl
      PREFIX nccl
      URL ${NCCL_URL}
      URL_MD5 ${NCCL_MD5}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE 1
      BUILD_COMMAND make -j${PROC_NUM} src.build CUDA_HOME=${CUDATOOLKIT_BIN_ROOT}
                    NVCC_GENCODE=${NCCL_GENCODE}
      INSTALL_COMMAND make src.install PREFIX=${NCCL_INSTALL_DIR}
      BUILD_BYPRODUCTS ${NCCL_LIBRARIES})

  endif(THIRD_PARTY)

endif()
