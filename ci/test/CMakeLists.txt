set(PYTHON_EXECUTABLE python3 CACHE STRING "python3 exe to run test, usually is the python3 installation oneflow is linked to")
set(ONEFLOW_SRC_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "source dir of oneflow")

file(GLOB_RECURSE PYTHON_TEST_FILES LIST_DIRECTORIES false "${ONEFLOW_SRC_DIR}/python/oneflow/test/test_*.py")
foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
  add_test(NAME ${PYTHON_TEST_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_TEST_FILE} --failfast --verbose
  )
  if(NOT BUILD_CUDA)
    set_tests_properties(
      ${PYTHON_TEST_FILE}
    PROPERTIES
      ENVIRONMENT "ONEFLOW_TEST_CPU_ONLY=1"
    )
  endif()
endforeach()
