# CTEST_OUTPUT_ON_FAILURE=1 CTEST_PARALLEL_LEVEL=20 ninja test
cmake_minimum_required(VERSION 3.19.0)

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
set(ONEFLOW_WHEEL_PATH /home/caishenghang/550ed7f9122589ab0f1ae89a50e0bf4702974895ddafdc9a10a77628d5387138/cu102/whl CACHE STRING "URL or local path for wheels")
enable_testing()
add_custom_target(ensure_docker_imgs)
function(add_docker_img)
  set(target_name "pull_docker_img_${ARGV0}")
  add_custom_target(${target_name})
  add_custom_command(TARGET ${target_name}
    COMMAND ${DOCKER_EXE} pull ${ARGV1}
  )
  add_dependencies(ensure_docker_imgs ${target_name})
endfunction()

add_custom_target(start_docker_containers)
function(add_docker_container)
  set(prefix ARG)
  set(singleValues "FIXTURE_NAME;IMG")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  set(fixtureName "${${prefix}_FIXTURE_NAME}")
  set(TEST_CONTAINER_NAME CI-$ENV{USER}-${fixtureName})
  set(${fixtureName}Name ${TEST_CONTAINER_NAME} PARENT_SCOPE)
  set(DOCKER_IMG_NAME "${${prefix}_IMG}")
  add_test(NAME Clear${fixtureName}
    COMMAND docker rm -f ${TEST_CONTAINER_NAME}
  )
  # start container and set as container fixture
  add_test(NAME Start${fixtureName}
    COMMAND docker run --name ${TEST_CONTAINER_NAME} --rm -d
      --privileged --shm-size=8g
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
      --runtime=nvidia
      -v /dataset:/dataset:ro -v /model_zoo:/model_zoo:ro
      -v ${ONEFLOW_WHEEL_PATH}:${ONEFLOW_WHEEL_PATH}:ro
      -v ${ONEFLOW_SRC_DIR}:${ONEFLOW_SRC_DIR}:ro
      -e ONEFLOW_WHEEL_PATH=${ONEFLOW_WHEEL_PATH}
      ${DOCKER_IMG_NAME}
      sleep 3600
  )
  set_tests_properties(Start${fixtureName} PROPERTIES
    DEPENDS Clear${fixtureName}
    FIXTURES_SETUP ${fixtureName}
    TIMEOUT 6000000000
  )
  # install pip and set as pip fixture
  add_test(NAME InstallOneFlowPipOn${fixtureName}
    COMMAND docker exec ${TEST_CONTAINER_NAME} python3 -m pip install --find-links=${ONEFLOW_WHEEL_PATH} oneflow
  )
  # stop container and set as container fixture cleanup
  add_test(NAME Stop${fixtureName}
    COMMAND docker rm -f ${TEST_CONTAINER_NAME}
  )
  set_tests_properties(Clear${fixtureName}               PROPERTIES FIXTURES_SETUP  ${fixtureName}  SKIP_RETURN_CODE TRUE)
  set_tests_properties(Start${fixtureName}               PROPERTIES FIXTURES_SETUP  ${fixtureName})
  set_tests_properties(InstallOneFlowPipOn${fixtureName} PROPERTIES FIXTURES_SETUP  ${fixtureName})
  set_tests_properties(Stop${fixtureName}                PROPERTIES FIXTURES_CLEANUP ${fixtureName} SKIP_RETURN_CODE TRUE)
endfunction()

set(TF_CONTAINER_FIXTURE "TFContainer")
set(TORCH_CONTAINER_FIXTURE "TorchContainer")
if(RUN_IN_DOCKER)
  set(DOCKER_EXE docker)
  add_docker_img(test_with_tf "${TEST_WITH_TF_IMG_TAG}")
  add_docker_img(test_with_torch "${TEST_WITH_TORCH_IMG_TAG}")
  add_docker_container(FIXTURE_NAME ${TF_CONTAINER_FIXTURE} IMG "${TEST_WITH_TF_IMG_TAG}")
  add_docker_container(FIXTURE_NAME ${TORCH_CONTAINER_FIXTURE} IMG "${TEST_WITH_TORCH_IMG_TAG}")
endif()

# TODO: ensure datasets (ExternalData)
# TODO: (if it is .whl URL) download pip package
# TODO: start docker, mount test dirs (writable), datasets (read-only)
# TODO: install pip package
function(add_oneflow_python_test)
  set(prefix ARG)
  set(testTypes "SINGLE_CLIENT;MULTI_CLIENT")
  list(APPEND noValues ${testTypes})
  set(singleValues "TEST_DIR")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  message(STATUS "Test dir: ${${prefix}_TEST_DIR}. Left-over args: ${ARG_UNPARSED_ARGUMENTS}. Keywords missing values: ${ARG_KEYWORDS_MISSING_VALUES}.")
  if ((NOT ${prefix}_SINGLE_CLIENT) AND (NOT ${prefix}_MULTI_CLIENT))
    message(FATAL_ERROR "must set one test type in ${testTypes}")
  endif()
  foreach(testType IN LISTS testTypes)
    if(${prefix}_${testType})
      list(APPEND testLabels ${testType})
      list(APPEND testNamePrefix ${testType})
    endif()
  endforeach()
  file(GLOB_RECURSE PYTHON_TEST_FILES LIST_DIRECTORIES false "${${prefix}_TEST_DIR}/test_*.py")
  foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
    set(TEST_NAME "[${testNamePrefix}] ${PYTHON_TEST_FILE}")
    if (RUN_IN_DOCKER)
      if (${prefix}_SINGLE_CLIENT)
        set(TEST_CONTAINER_NAME "${${TF_CONTAINER_FIXTURE}Name}")
      endif()
      if (${prefix}_MULTI_CLIENT)
        set(TEST_CONTAINER_NAME "${${TORCH_CONTAINER_FIXTURE}Name}")
      endif()
      set(DOCKER_EXEC_PREFIX "docker;exec;${TEST_CONTAINER_NAME}")
    endif()
    # use sh -c to prevent DOCKER_EXEC_PREFIX is treated as one command
    add_test(NAME ${TEST_NAME}
      COMMAND ${DOCKER_EXEC_PREFIX};${PYTHON_EXECUTABLE};${PYTHON_TEST_FILE};--failfast;--verbose
    )
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS ${testLabels})
    set_tests_properties(${TEST_NAME}
      PROPERTIES
        ENVIRONMENT "$<$<NOT:$<BOOL:${BUILD_CUDA}>>:ONEFLOW_TEST_CPU_ONLY=1>;$<$<BOOL:${IS_DEV}>:PYTHONPATH=${ONEFLOW_SRC_DIR}/python:$ENV{PYTHONPATH}>"
        RESOURCE_GROUPS
          "vram:2000"
    )
    if (RUN_IN_DOCKER)
      if (${prefix}_SINGLE_CLIENT)
        set_tests_properties(${TEST_NAME} PROPERTIES FIXTURES_REQUIRED "${TF_CONTAINER_FIXTURE};OneFlowPipOn${TF_CONTAINER_FIXTURE}")
      endif()
      if (${prefix}_MULTI_CLIENT)
        set_tests_properties(${TEST_NAME} PROPERTIES FIXTURES_REQUIRED "${TORCH_CONTAINER_FIXTURE};OneFlowPipOn${TORCH_CONTAINER_FIXTURE}")
      endif()
    endif()
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_rnn.py$")
      set_tests_properties(${TEST_NAME}
          PROPERTIES
            RESOURCE_GROUPS
              "vram:4000"
      )
    endif()
  endforeach()
endfunction()
add_oneflow_python_test(TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/modules MULTI_CLIENT)
