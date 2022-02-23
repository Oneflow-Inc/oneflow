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
set(BUILD_CUDA ON CACHE BOOL "")
# TODO: generate resource spec by running nvidia-smi
set(CODEGEN_PYTHON_EXECUTABLE python3 CACHE STRING "Python executable to generate spec .json files")
set(CTEST_RESOURCE_SPEC_FILE "${PROJECT_BINARY_DIR}/gen-resource-spec.json" CACHE STRING "" FORCE)
if(BUILD_CUDA)
  set(CTEST_RESOURCE_SPEC_FILE_ARGS "--cuda")
endif()
message(STATUS "ONEFLOW_SRC_DIR: ${ONEFLOW_SRC_DIR}")
execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/gen-resource-spec.py ${CTEST_RESOURCE_SPEC_FILE_ARGS}
  OUTPUT_FILE ${CTEST_RESOURCE_SPEC_FILE}
  COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

enable_testing()
if(EXISTS ${CTEST_RESOURCE_SPEC_FILE})
  message(STATUS "using CTEST_RESOURCE_SPEC_FILE: ${CTEST_RESOURCE_SPEC_FILE}")
else()
  message(FATAL_ERROR "CTEST_RESOURCE_SPEC_FILE not found: ${CTEST_RESOURCE_SPEC_FILE}")
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
  string(RANDOM LENGTH 1 ALPHABET "123" leftNumbersSize)
  string(RANDOM LENGTH ${leftNumbersSize} ALPHABET "0123456789" leftNumbers)
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
    set(DISTRIBUTED_LAUNCH_PREFIX "")
    if (NOT ${prefix}_SINGLE_CLIENT)
      get_random_port(RESULT portNumber)
      if(ONEFLOW_TEST_DEVICE_NUM GREATER 1)
        set(DISTRIBUTED_LAUNCH_PREFIX "-m;oneflow.distributed.launch;--nproc_per_node;${ONEFLOW_TEST_DEVICE_NUM};--master_port;${portNumber}")
        set(portNumberPrefix "[port ${portNumber}]")
      endif()
    endif()
    set(TEST_NAME "[GPUx${ONEFLOW_TEST_DEVICE_NUM}]${testNamePrefix}${portNumberPrefix}[${${prefix}_NAME}]${PYTHON_TEST_FILE}")
    # use a list to prevent it is treated as one command
    add_test(NAME ${TEST_NAME}
      COMMAND ${PYTHON_EXECUTABLE};${DISTRIBUTED_LAUNCH_PREFIX};${PYTHON_TEST_FILE};--failfast;--verbose
    )
    set(vramRequired 600)
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_resnet50|python/oneflow/test/modules/test_compatiblity.py")
      set(vramRequired 6000)
    endif()
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_rnn.py$")
      set(vramRequired 4000)
    endif()
    if(PYTHON_TEST_FILE MATCHES "python/oneflow/test/modules/test_convtranspose.py$|modules/test_conv2d.py|modules/test_loss.py")
      set(vramRequired 3000)
    endif()
    if(PYTHON_TEST_FILE MATCHES "modules/test_ddp_multi_outputs.py$|modules/test_min_max_observer.py|modules/test_module.py")
      set(vramRequired 1000)
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
    set(PROXY_ENV_VAR "HTTP_PROXY=;HTTPS_PROXY=;http_proxy=;https_proxy=")
    set(resourceGroups "${resourceGroups},cpu:${ONEFLOW_TEST_DEVICE_NUM}")
    set_tests_properties(${TEST_NAME}
      PROPERTIES
        ENVIRONMENT "${PROXY_ENV_VAR};$<$<NOT:$<BOOL:${BUILD_CUDA}>>:ONEFLOW_TEST_CPU_ONLY=1>;ONEFLOW_TEST_DEVICE_NUM=${ONEFLOW_TEST_DEVICE_NUM}"
        RESOURCE_GROUPS "${resourceGroups}"
        LABELS "${testLabels};"
        TIMEOUT 300 # 5 minutes
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
