# - Try to find Nvidia NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:            Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#  NCCL_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(NCCL_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIR nccl.h
    PATHS ${NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

find_library(NCCL_LIBRARY libnccl_static.a
    PATHS ${NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(
    NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
	# get NCCL version
  file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_HEADER_CONTENTS)
	string(REGEX MATCH "define NCCL_MAJOR * +([0-9]+)"
				 NCCL_VERSION_MAJOR "${NCCL_HEADER_CONTENTS}")
	string(REGEX REPLACE "define NCCL_MAJOR * +([0-9]+)" "\\1"
				 NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
	string(REGEX MATCH "define NCCL_MINOR * +([0-9]+)"
				 NCCL_VERSION_MINOR "${NCCL_HEADER_CONTENTS}")
	string(REGEX REPLACE "define NCCL_MINOR * +([0-9]+)" "\\1"
				 NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
	string(REGEX MATCH "define NCCL_PATCH * +([0-9]+)"
				 NCCL_VERSION_PATCH "${NCCL_HEADER_CONTENTS}")
	string(REGEX REPLACE "define NCCL_PATCH * +([0-9]+)" "\\1"
				 NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
  # Assemble NCCL version
  if(NOT NCCL_VERSION_MAJOR)
    set(NCCL_VERSION "?")
  else()
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
  endif()

  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  message(STATUS "Found NCCL: v${NCCL_VERSION}  (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARY})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_LIBRARY NCCL_INCLUDE_DIR)
endif()
    
