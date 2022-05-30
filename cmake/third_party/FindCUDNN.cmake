# - Try to find cuDNN
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT_DIR:            Base directory where all cuDNN components are found
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_DIRS
#  CUDNN_LIBRARIES
#  CUDNN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)
include(CMakeDependentOption)

set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

if(OF_CUDA_LINK_DYNAMIC_LIBRARY)
  set(CUDNN_STATIC OFF)
endif()
set(__cudnn_libname "libcudnn.so")

find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS}
          PATH_SUFFIXES cuda/include include)

unset(CUDNN_LIBRARY CACHE)
find_library(CUDNN_LIBRARY ${__cudnn_libname} HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR}
             PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(CUDNN_FOUND)
  # get cuDNN version
  if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_HEADER_CONTENTS)
  else()
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
  endif()
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)" CUDNN_VERSION_MAJOR
               "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1" CUDNN_VERSION_MAJOR
                       "${CUDNN_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)" CUDNN_VERSION_MINOR
               "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1" CUDNN_VERSION_MINOR
                       "${CUDNN_VERSION_MINOR}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)" CUDNN_VERSION_PATCH
               "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1" CUDNN_VERSION_PATCH
                       "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()

  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})

  get_filename_component(CUDNN_LIBRARY_DIRECTORY ${CUDNN_LIBRARY} DIRECTORY)
  if(NOT CUDNN_STATIC)
    # skipping: libcudnn_adv_infer.so libcudnn_adv_train.so
    set(CUDNN_DYNAMIC_NAMES libcudnn_cnn_infer.so libcudnn_cnn_train.so libcudnn_ops_infer.so
                            libcudnn_ops_train.so)
    foreach(CUDNN_DYNAMIC_NAME ${CUDNN_DYNAMIC_NAMES})
      list(APPEND CUDNN_LIBRARIES ${CUDNN_LIBRARY_DIRECTORY}/${CUDNN_DYNAMIC_NAME})
    endforeach()
  else()
    set(SINGLE_CUDNN_STATIC_LIB ${CUDNN_LIBRARY_DIRECTORY}/libcudnn_static.a)
    if(EXISTS ${SINGLE_CUDNN_STATIC_LIB})
      get_filename_component(SINGLE_CUDNN_STATIC_LIB "${SINGLE_CUDNN_STATIC_LIB}" REALPATH)
      list(APPEND CUDNN_LIBRARIES ${SINGLE_CUDNN_STATIC_LIB})
    else()
      list(
        APPEND
        CUDNN_LIBRARIES
        # ${CUDNN_LIBRARY_DIRECTORY}/libcudnn_ops_infer_static.a
        # ${CUDNN_LIBRARY_DIRECTORY}/libcudnn_ops_train_static.a
        ${CUDNN_LIBRARY_DIRECTORY}/libcudnn_cnn_infer_static.a
        ${CUDNN_LIBRARY_DIRECTORY}/libcudnn_cnn_train_static.a)
    endif()
    list(APPEND CUDNN_LIBRARIES CUDA::cublas_static CUDA::cublasLt_static)
    # Starting cudnn 8.3, zlib should be linked additionally
    # For more info: https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html
    if(CUDNN_VERSION VERSION_GREATER_EQUAL "8.3")
      list(APPEND CUDNN_LIBRARIES zlib_imported)
    endif()
    if(CUDNN_WHOLE_ARCHIVE)
      list(PREPEND CUDNN_LIBRARIES -Wl,--whole-archive)
      list(APPEND CUDNN_LIBRARIES -Wl,--no-whole-archive)
    endif()
  endif()
  message(
    STATUS
      "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARIES})"
  )
  mark_as_advanced(CUDNN_ROOT_DIR CUDNN_LIBRARY CUDNN_INCLUDE_DIR)
endif()
