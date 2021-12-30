# CTEST_OUTPUT_ON_FAILURE=1 CTEST_PARALLEL_LEVEL=20 ninja testcmake_minimum_required(VERSION 3.19.0)

project(oneflow-ci-test)
include(CMakeDependentOption)
set(PYTHON_EXECUTABLE python3 CACHE STRING "python3 exe to run test, usually is the python3 installation oneflow is linked to")
set(ONEFLOW_SRC_DIR ${CMAKE_SOURCE_DIR} CACHE STRING "source dir of oneflow")
set(IS_DEV ON CACHE BOOL "")
set(BUILD_CUDA ON CACHE BOOL "")
set(RUN_IN_DOCKER ON CACHE BOOL "")
set(PULL_DOCKER_IMG ON CACHE BOOL "")
set(TEST_WITH_TF_IMG_TAG "registry.cn-beijing.aliyuncs.com/oneflow/test-with-tf-2.3.0:2f831e9354298a11447578e869d983959feb046f" CACHE STRING "")
set(TEST_WITH_TORCH_IMG_TAG "registry.cn-beijing.aliyuncs.com/oneflow/test-with-pytorch-1.9.0:e7a497b41d8b7f1bce055b1f23d027f93b1557ae" CACHE STRING "")
set(CTEST_RESOURCE_SPEC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resource-spec/2x-rtx-2080.json" CACHE STRING "")
set(SINGLE_CLIENT_TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test CACHE STRING "test script dir for single client")
set(MULTI_CLIENT_TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test CACHE STRING "test script dir for multi client")

add_custom_target(ensure_docker_imgs)
function(add_docker_img)
  set(target_name "pull_docker_img_${ARGV0}")
  add_custom_target(${target_name})
  add_custom_command(TARGET ${target_name}
    COMMAND ${DOCKER_EXE} pull ${ARGV1}
  )
  add_dependencies(ensure_docker_imgs ${target_name})
endfunction()

if(RUN_IN_DOCKER)
  set(DOCKER_EXE docker)
  add_docker_img(test_with_tf "${TEST_WITH_TF_IMG_TAG}")
  add_docker_img(test_with_torch "${TEST_WITH_TORCH_IMG_TAG}")
endif()

# TODO: ensure datasets (ExternalData)
# TODO: start docker, mount test dirs (writable), datasets (read-only)
function(add_oneflow_python_test)
set(prefix ARG)
set(singleValues "TEST_DIR")
cmake_parse_arguments(
  PARSE_ARGV 0
  ${prefix}
  "${noValues}" "${singleValues}" "${multiValues}"
)
file(GLOB_RECURSE PYTHON_TEST_FILES LIST_DIRECTORIES false "${ARG_TEST_DIR}/test_*.py")
foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
  set(TEST_NAME ${PYTHON_TEST_FILE})
  add_test(NAME ${TEST_NAME}
    COMMAND ${PYTHON_EXECUTABLE} ${ONEFLOW_SRC_DIR}/${PYTHON_TEST_FILE} --failfast --verbose
  )
  set_tests_properties(${TEST_NAME}
    PROPERTIES
      ENVIRONMENT "$<$<NOT:$<BOOL:${BUILD_CUDA}>>:ONEFLOW_TEST_CPU_ONLY=1>;$<$<BOOL:${IS_DEV}>:PYTHONPATH=${ONEFLOW_SRC_DIR}/python:$ENV{PYTHONPATH}>"
      RESOURCE_GROUPS
        "vram:2000"
  )
endforeach()
endfunction()

set_tests_properties(python/oneflow/test/modules/test_rnn.py
  PROPERTIES
    RESOURCE_GROUPS
      "vram:4000"
)
