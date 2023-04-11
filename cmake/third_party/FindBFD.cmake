# - BFD Library module.
#=============================================================================
# This module finds libbfd and associated headers.
#
#=== Variables ===============================================================
# This module will set the following variables in your project:
#
#   BFD_FOUND            Whether libbfd was successfully found.
#   bfd::bfd             Cmake target for bfd
#
#=============================================================================

include(FindPackageHandleStandardArgs)

set(CMAKE_LIBRARY_PATH /lib /usr/lib /usr/local/lib)
set(CMAKE_INCLUDE_PATH /usr/include /usr/local/include)

find_path(BFD_INCLUDE_PATH bfd.h PATH /usr/include /usr/local/include)
find_library(BFD_LIBRARIES bfd PATH /lib /usr/lib /usr/local/lib)

find_package_handle_standard_args(BFD DEFAULT_MSG BFD_LIBRARIES BFD_INCLUDE_PATH)

if(BFD_FOUND)
  if(NOT TARGET bfd::bfd)
    add_library(bfd::bfd INTERFACE IMPORTED)
    set_property(TARGET bfd::bfd PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BFD_INCLUDE_PATH})
    set_property(TARGET bfd::bfd PROPERTY INTERFACE_LINK_LIBRARIES ${BFD_LIBRARIES})
    set_property(TARGET bfd::bfd PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  endif(NOT TARGET bfd::bfd)
endif()

mark_as_advanced(BFD_INCLUDE_PATH BFD_LIBRARIES)
