if(NOT DEFINED Python3_EXECUTABLE)
  execute_process(
    COMMAND which python3
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE OUTPUT
    ERROR_QUIET)
  if(STATUS EQUAL 0)
    string(STRIP ${OUTPUT} STRIPPED)
    message(STATUS "Using Python3 from 'which python3': ${STRIPPED}")
    set(Python3_EXECUTABLE ${STRIPPED})
  endif()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Python3 specified. Version found: " ${Python3_VERSION})
set(Python_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "Using Python executable: " ${Python_EXECUTABLE})

message(STATUS "Installing necessary Python packages...")
set(requirements_txt ${PROJECT_SOURCE_DIR}/dev-requirements.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${requirements_txt})
message(STATUS "PIP_INDEX_MIRROR: ${PIP_INDEX_MIRROR}")
if(PIP_INDEX_MIRROR)
  set(extra_index_arg "-i")
endif()

function(install_py_dev_deps)
  execute_process(COMMAND ${ARGV0} -m pip install ${extra_index_arg} ${PIP_INDEX_MIRROR} -r
                          ${requirements_txt} --user RESULT_VARIABLE PIP_INSTALL_STATUS)
  if(NOT PIP_INSTALL_STATUS EQUAL 0)
    message(FATAL_ERROR "fail to install pip packages")
  endif()
  message(STATUS "Python packages are installed.")
endfunction(install_py_dev_deps)
install_py_dev_deps(${Python_EXECUTABLE})

find_package(Python3 COMPONENTS Development NumPy)
if(Python3_Development_FOUND AND Python3_INCLUDE_DIRS)
  set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
endif()
if(Python3_NumPy_FOUND AND Python3_NumPy_INCLUDE_DIRS)
  set(Python_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})
endif()
if(NOT Python_INCLUDE_DIRS)
  message(STATUS "Getting python include directory from sysconfig..")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
    OUTPUT_VARIABLE Python_INCLUDE_DIRS RESULT_VARIABLE ret_code)
  string(STRIP "${Python_INCLUDE_DIRS}" Python_INCLUDE_DIRS)
  if((NOT (ret_code EQUAL "0")) OR (NOT IS_DIRECTORY ${Python_INCLUDE_DIRS})
     OR (NOT EXISTS ${Python_INCLUDE_DIRS}/Python.h))
    set(Python_INCLUDE_DIRS "")
  endif()
endif()
if(NOT Python_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find python include directory")
endif()
message(STATUS "Found python include directory ${Python_INCLUDE_DIRS}")

if(NOT Python_NumPy_INCLUDE_DIRS)
  message(STATUS "Getting numpy include directory by numpy.get_include()..")
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                  OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS RESULT_VARIABLE ret_code)
  string(STRIP "${Python_NumPy_INCLUDE_DIRS}" Python_NumPy_INCLUDE_DIRS)
  if((NOT ret_code EQUAL 0) OR (NOT IS_DIRECTORY ${Python_NumPy_INCLUDE_DIRS})
     OR (NOT EXISTS ${Python_NumPy_INCLUDE_DIRS}/numpy/arrayobject.h))
    set(Python_NumPy_INCLUDE_DIRS "")
  endif()
endif()
if(NOT Python_NumPy_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find numpy include directory")
endif()
message(STATUS "Found numpy include directory ${Python_NumPy_INCLUDE_DIRS}")

# PYTHON_EXECUTABLE will be used by pybind11
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
include(pybind11)

set(CODEGEN_PYTHON_EXECUTABLE ${Python_EXECUTABLE}
    CACHE STRING "Python executable to generate .cpp/.h files")
if(NOT "${CODEGEN_PYTHON_EXECUTABLE}" STREQUAL "${Python_EXECUTABLE}")
  install_py_dev_deps(${CODEGEN_PYTHON_EXECUTABLE})
endif()
