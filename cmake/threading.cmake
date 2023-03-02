foreach(threading_runtime_item ${CPU_THREADING_RUNTIMES})
  if(NOT ${threading_runtime_item} MATCHES "^(TBB|OMP)$")
    message(FATAL_ERROR "Unsupported cpu threading runtime: ${threading_runtime_item}")
  endif()

  if(${threading_runtime_item} STREQUAL "OMP")
    # Reference:
    # https://releases.llvm.org/11.0.0/tools/clang/docs/OpenMPSupport.html
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 11)
        message(
          FATAL_ERROR
            "libopenmp is not supported under clang10, please use TBB with '-DCPU_THREADING_RUNTIMES=TBB'."
        )
      endif()
    endif()
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(WITH_${threading_runtime_item} ON)
      add_definitions(-DWITH_${threading_runtime_item})
    endif()
  else()
    set(WITH_${threading_runtime_item} ON)
    add_definitions(-DWITH_${threading_runtime_item})
  endif()
endforeach()
