if(WIN32)
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC
                  -D__VERSION__=\"MSVC\")
  add_definitions(
    -DWIN32
    -DOS_WIN
    -D_MBCS
    -DWIN64
    -DWIN32_LEAN_AND_MEAN
    -DNOGDI
    -DPLATFORM_WINDOWS
    -D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(
    /bigobj
    /nologo
    /EHsc
    /GF
    /FC
    /MP
    /Gm-)
  add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  foreach(
    flag_var
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()

  # set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS} /DEBUG:FASTLINK")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=0")
else()
  set(EXTRA_CXX_FLAGS "-Wall -Wno-sign-compare -Wno-unused-function -fPIC")

  if(APPLE)
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wno-deprecated-declarations")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_CXX_FLAGS}")
endif(WIN32)
