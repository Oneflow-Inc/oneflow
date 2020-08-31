find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "Python3 specified. Version found: " ${Python3_VERSION})
set(Python_EXECUTABLE ${Python3_EXECUTABLE})
message(STATUS "Using Python executable: " ${Python_EXECUTABLE})

message(STATUS "Installing necessary Python packages...")
set(requirements_txt ${PROJECT_SOURCE_DIR}/dev-requirements.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${requirements_txt})
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pip install -r ${requirements_txt} --user
)
message(STATUS "Python packages are installed.")

find_package(Python3 COMPONENTS Development NumPy)
if (Python3_Development_FOUND AND Python3_INCLUDE_DIRS)
  set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
endif()
if (Python3_NumPy_FOUND AND Python3_NumPy_INCLUDE_DIRS)
  set(Python_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})
endif()
if (NOT Python_INCLUDE_DIRS)
  message(STATUS "Getting python include directory from sysconfig..")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])"
    OUTPUT_VARIABLE Python_INCLUDE_DIRS
    RESULT_VARIABLE ret_code)
  string(STRIP "${Python_INCLUDE_DIRS}" Python_INCLUDE_DIRS)
  if ((NOT (ret_code EQUAL "0")) OR (NOT IS_DIRECTORY ${Python_INCLUDE_DIRS})
    OR (NOT EXISTS ${Python_INCLUDE_DIRS}/Python.h))
    set(Python_INCLUDE_DIRS "")
  endif()
endif()
if (NOT Python_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find python include directory")
endif()
message(STATUS "Found python include directory ${Python_INCLUDE_DIRS}")

if (NOT Python_NumPy_INCLUDE_DIRS)
  message(STATUS "Getting numpy include directory by numpy.get_include()..")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS
    RESULT_VARIABLE ret_code)
  string(STRIP "${Python_NumPy_INCLUDE_DIRS}" Python_NumPy_INCLUDE_DIRS)
  if ((NOT ret_code EQUAL 0) OR (NOT IS_DIRECTORY ${Python_NumPy_INCLUDE_DIRS})
    OR (NOT EXISTS ${Python_NumPy_INCLUDE_DIRS}/numpy/arrayobject.h))
    set(Python_NumPy_INCLUDE_DIRS "")
  endif()
endif()
if (NOT Python_NumPy_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find numpy include directory")
endif()
message(STATUS "Found numpy include directory ${Python_NumPy_INCLUDE_DIRS}")

# PYTHON_EXECUTABLE will be used by pybind11
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
include(pybind11)
