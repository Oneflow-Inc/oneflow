set(PYTHON_EXECUTABLE python3 CACHE STRING "python3 exe to run test, usually is the python3 installation oneflow is linked to")
set(ONEFLOW_SRC_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "source dir of oneflow")
set(IS_DEV ON CACHE BOOL "")

file(GLOB_RECURSE PYTHON_TEST_FILES LIST_DIRECTORIES false "${ONEFLOW_SRC_DIR}/python/oneflow/test_*.py")
foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
  set(TEST_NAME ${PYTHON_TEST_FILE})
  add_test(NAME ${TEST_NAME}
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_TEST_FILE} --failfast --verbose
  )
  set_tests_properties(${TEST_NAME}
    PROPERTIES
      ENVIRONMENT "$<$<NOT:$<BOOL:${BUILD_CUDA}>>:ONEFLOW_TEST_CPU_ONLY=1>;$<$<BOOL:${IS_DEV}>:PYTHONPATH=${ONEFLOW_SRC_DIR}/python:$ENV{PYTHONPATH}>"
  )
endforeach()
