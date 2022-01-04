# CTEST_OUTPUT_ON_FAILURE=1 CTEST_PARALLEL_LEVEL=$(nproc) ninja test
cmake_minimum_required(VERSION 3.19.0)
find_program(CTEST_EXECUTABLE NAMES ctest REQUIRED)
execute_process(
  COMMAND ${CTEST_EXECUTABLE} --version
  OUTPUT_VARIABLE CTEST_VERSION
  COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "([0-9].+[0-9])" CTEST_VERSION ${CTEST_VERSION})
set(MINIMUM_CTEST_VERSION 3.16.0)
if (CTEST_VERSION VERSION_LESS ${MINIMUM_CTEST_VERSION})
  message(FATAL_ERROR "OneFlow CI test uses RESOURCE_GROUPS which requires ctest >= ${MINIMUM_CTEST_VERSION}, found ctest: ${CTEST_EXECUTABLE}, version: ${CTEST_VERSION}")
endif()
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
# TODO: generate resource spec by running nvidia-smi
set(CODEGEN_PYTHON_EXECUTABLE python3 CACHE STRING "Python executable to generate spec .json files")
set(CTEST_RESOURCE_SPEC_FILE "${PROJECT_BINARY_DIR}/gen-resource-spec.json" CACHE STRING "" FORCE)
if(BUILD_CUDA)
  set(CTEST_RESOURCE_SPEC_FILE_ARGS "--cuda")
endif()

execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/gen-resource-spec.py ${CTEST_RESOURCE_SPEC_FILE_ARGS}
  OUTPUT_FILE ${CTEST_RESOURCE_SPEC_FILE}
  COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(ONEFLOW_WHEEL_PATH /home/caishenghang/550ed7f9122589ab0f1ae89a50e0bf4702974895ddafdc9a10a77628d5387138/cu102/whl CACHE STRING "URL or local path for wheels")
enable_testing()
if(EXISTS ${CTEST_RESOURCE_SPEC_FILE})
  message(STATUS "using CTEST_RESOURCE_SPEC_FILE: ${CTEST_RESOURCE_SPEC_FILE}")
else()
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE not found: ${CTEST_RESOURCE_SPEC_FILE}")
endif()
function(add_docker_container)
  set(prefix ARG)
  set(singleValues "FIXTURE_NAME;IMG")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  set(fixtureName "${${prefix}_FIXTURE_NAME}")
  string(RANDOM LENGTH 8 ALPHABET "0123456789asdfghjklqwertyuiopzxcvbnm" dockerNameRandomSuffix)
  set(TEST_CONTAINER_NAME CI-$ENV{USER}-${fixtureName}-${dockerNameRandomSuffix})
  set(${fixtureName}ContainerName ${TEST_CONTAINER_NAME} PARENT_SCOPE)
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
  )
  # install pip and set as pip fixture
  add_test(NAME InstallOneFlowPipOn${fixtureName}
    COMMAND docker exec ${TEST_CONTAINER_NAME} python3 -m pip install --find-links=${ONEFLOW_WHEEL_PATH} oneflow
  )
  # stop container and set as container fixture cleanup
  add_test(NAME Stop${fixtureName}
    COMMAND docker rm -f ${TEST_CONTAINER_NAME}
  )
  set_tests_properties(Clear${fixtureName}               PROPERTIES  FIXTURES_SETUP  ${fixtureName} SKIP_RETURN_CODE 1)
  set_tests_properties(Start${fixtureName}               PROPERTIES  FIXTURES_SETUP  ${fixtureName} DEPENDS Clear${fixtureName})
  set_tests_properties(InstallOneFlowPipOn${fixtureName} PROPERTIES  FIXTURES_SETUP  ${fixtureName} DEPENDS Start${fixtureName})
  set_tests_properties(Stop${fixtureName}                PROPERTIES FIXTURES_CLEANUP ${fixtureName} SKIP_RETURN_CODE 1)
endfunction()

set(TF_CONTAINER_FIXTURE "TFContainer")
set(TORCH_CONTAINER_FIXTURE "TorchContainer")
if(RUN_IN_DOCKER)
  set(CTEST_DOCKER_EXEC ${PROJECT_SOURCE_DIR}/bin/ctest-docker-exec)
endif()

# Start docker
if (RUN_IN_DOCKER)
  add_docker_container(FIXTURE_NAME ${TF_CONTAINER_FIXTURE} IMG "${TEST_WITH_TF_IMG_TAG}")
  add_docker_container(FIXTURE_NAME ${TORCH_CONTAINER_FIXTURE} IMG "${TEST_WITH_TORCH_IMG_TAG}")
endif()

function(get_combinations_of_two)
  set(prefix ARG)
  set(singleValues "PREFIX;SUFFIX;ITEMS;RESULT")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  foreach(i IN LISTS ${prefix}_ITEMS)
    foreach(j IN LISTS ${prefix}_ITEMS)
      if(i LESS j)
        set(iElement ${${prefix}_PREFIX}${i}${${prefix}_SUFFIX})
        set(jElement ${${prefix}_PREFIX}${j}${${prefix}_SUFFIX})
        list(APPEND result "${iElement},${jElement}")
      endif()
    endforeach()
  endforeach()
  set(${${prefix}_RESULT} ${result} PARENT_SCOPE)
endfunction()

function(get_random_port)
  set(prefix ARG)
  set(singleValues "RESULT")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  # max port number 65,535
  string(RANDOM LENGTH 1 ALPHABET "123456" firstNumber)
  string(RANDOM LENGTH 1 ALPHABET "01234" secondNumber) # max is 4 to avoid overflow
  string(RANDOM LENGTH 3 ALPHABET "0123456789" leftNumbers)
  set(${${prefix}_RESULT} ${firstNumber}${secondNumber}${leftNumbers} PARENT_SCOPE)
endfunction()

# TODO: ensure datasets (ExternalData)
# TODO: (if it is .whl URL) download pip package
function(add_oneflow_python_test)
  set(prefix ARG)
  set(testTypes "SINGLE_CLIENT")
  list(APPEND noValues ${testTypes})
  set(singleValues "TEST_DIR;NAME")
  cmake_parse_arguments(
    PARSE_ARGV 0
    ${prefix}
    "${noValues}" "${singleValues}" "${multiValues}"
  )
  message(DEBUG "Test dir: ${${prefix}_TEST_DIR}. Left-over args: ${ARG_UNPARSED_ARGUMENTS}. Keywords missing values: ${ARG_KEYWORDS_MISSING_VALUES}.")
  if (${prefix}_SINGLE_CLIENT)
    set(testNamePrefix "[legacy]")
    list(APPEND testLabels SingleClient)
  endif()
  # Glob Python scripts
  file(GLOB_RECURSE PYTHON_TEST_FILES LIST_DIRECTORIES false "${${prefix}_TEST_DIR}/test_*.py")
  foreach(PYTHON_TEST_FILE ${PYTHON_TEST_FILES})
  foreach(ONEFLOW_TEST_DEVICE_NUM 1;2;4)
    if (NOT ${prefix}_SINGLE_CLIENT)
      get_random_port(RESULT portNumber)
      set(DISTRIBUTED_LAUNCH_PREFIX "-m;oneflow.distributed.launch;--nproc_per_node;${ONEFLOW_TEST_DEVICE_NUM};--master_port;${portNumber}")
      set(portNumberPrefix "[port ${portNumber}]")
    endif()
    set(TEST_NAME "[GPUx${ONEFLOW_TEST_DEVICE_NUM}]${testNamePrefix}${portNumberPrefix}[${${prefix}_NAME}]${PYTHON_TEST_FILE}")
    set(DOCKER_EXEC_PREFIX "${CTEST_DOCKER_EXEC}")
    if (RUN_IN_DOCKER)
      set(CONTAINER_FIXTURE ${TORCH_CONTAINER_FIXTURE})
      if (${prefix}_SINGLE_CLIENT)
        set(CONTAINER_FIXTURE "${TF_CONTAINER_FIXTURE}")
      endif()
      set(TEST_CONTAINER_NAME "${${CONTAINER_FIXTURE}ContainerName}")
      if (NOT TEST_CONTAINER_NAME)
        message(FATAL_ERROR "found empty TEST_CONTAINER_NAME: ${TEST_CONTAINER_NAME}")
      endif()
      list(APPEND DOCKER_EXEC_PREFIX ${TEST_CONTAINER_NAME})
    endif()
    # use a list to prevent DOCKER_EXEC_PREFIX is treated as one command
    add_test(NAME ${TEST_NAME}
      COMMAND ${DOCKER_EXEC_PREFIX};${PYTHON_EXECUTABLE};${DISTRIBUTED_LAUNCH_PREFIX};${PYTHON_TEST_FILE};--failfast;--verbose
    )
    set(vramRequired 600)
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_rnn.py$")
      set(vramRequired 4000)
    endif()
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_convtranspose.py$")
      set(vramRequired 3000)
    endif()
    if(ONEFLOW_TEST_DEVICE_NUM EQUAL 1)
      set(resourceGroups "cuda0:${vramRequired};cuda1:${vramRequired};cuda2:${vramRequired};cuda3:${vramRequired}")
      string(RANDOM LENGTH 1 ALPHABET "0123" randomIdx)
      list(GET resourceGroups ${randomIdx} resourceGroups)
    elseif(ONEFLOW_TEST_DEVICE_NUM EQUAL 2)
      string(RANDOM LENGTH 1 ALPHABET "012345" randomIdx)
      get_combinations_of_two(PREFIX cuda SUFFIX ":${vramRequired}" ITEMS "0;1;2;3" RESULT resourceGroups)
      list(GET resourceGroups ${randomIdx} resourceGroups)
    elseif(ONEFLOW_TEST_DEVICE_NUM EQUAL 4)
      set(resourceGroups "cuda0:${vramRequired},cuda1:${vramRequired},cuda2:${vramRequired},cuda3:${vramRequired}")
    else()
      message(FATAL_ERROR "unsupprted ONEFLOW_TEST_DEVICE_NUM: ${ONEFLOW_TEST_DEVICE_NUM}")
    endif()
    set(resourceGroups "${resourceGroups},cpu:${ONEFLOW_TEST_DEVICE_NUM}")
    set_tests_properties(${TEST_NAME}
      PROPERTIES
        ENVIRONMENT "$<$<BOOL:${RUN_IN_DOCKER}>:RUN_IN_DOCKER=1>;$<$<NOT:$<BOOL:${BUILD_CUDA}>>:ONEFLOW_TEST_CPU_ONLY=1>;$<$<BOOL:${IS_DEV}>:PYTHONPATH=${ONEFLOW_SRC_DIR}/python:$ENV{PYTHONPATH}>;ONEFLOW_TEST_DEVICE_NUM=${ONEFLOW_TEST_DEVICE_NUM}"
        RESOURCE_GROUPS "${resourceGroups}"
        LABELS "${testLabels};"
        FIXTURES_REQUIRED ${CONTAINER_FIXTURE}
    )
  endforeach()
  endforeach()
endfunction()
add_oneflow_python_test(NAME module TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/modules)
add_oneflow_python_test(NAME dataloader TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/dataloader)
add_oneflow_python_test(NAME graph TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/graph)
add_oneflow_python_test(NAME tensor TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/tensor)
add_oneflow_python_test(NAME xrt TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/test/xrt)
add_oneflow_python_test(NAME ir TEST_DIR ${ONEFLOW_SRC_DIR}/oneflow/ir/test)
add_oneflow_python_test(NAME op TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test/ops SINGLE_CLIENT)
add_oneflow_python_test(NAME custom_op TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test/custom_ops SINGLE_CLIENT)
add_oneflow_python_test(NAME serving TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test/serving SINGLE_CLIENT)
add_oneflow_python_test(NAME model TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test/models SINGLE_CLIENT)
add_oneflow_python_test(NAME xrt TEST_DIR ${ONEFLOW_SRC_DIR}/python/oneflow/compatible/single_client/test/xrt SINGLE_CLIENT)
