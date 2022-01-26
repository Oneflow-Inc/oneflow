include(ExternalProject)

if(WITH_OPENVINO)

  set(OPENVINO_INCLUDE_DIR "")
  set(OPENVINO_LIBRARIES "")

  find_path(
    OPENVINO_INFERENCE_INCLUDE_DIR inference_engine.hpp
    PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/deployment_tools/inference_engine/include
          $ENV{OPENVINO_ROOT} $ENV{OPENVINO_ROOT}/ideployment_tools/inference_engine/include
          ${THIRD_PARTY_DIR}/OpenVINO/deployment_tools/inference_engine/include)

  find_path(
    OPENVINO_NGRAPH_INCLUDE_DIR
    ngraph/op/add.hpp
    ${OPENVINO_ROOT}
    ${OPENVINO_ROOT}/deployment_tools/ngraph/include
    $ENV{OPENVINO_ROOT}
    $ENV{OPENVINO_ROOT}/ideployment_tools/ngraph/include
    ${THIRD_PARTY_DIR}/OpenVINO/deployment_tools/ngraph/include)

  find_library(
    OPENVINO_INFERENCE_LIBRARIES
    NAMES libinference_engine.so libinference_engine.a
    PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/deployment_tools/inference_engine/lib/intel64/
          $ENV{OPENVINO_ROOT} $ENV{OPENVINO_ROOT}/deployment_tools/inference_engine/lib/intel64/
          ${THIRD_PARTY_DIR}/OPENVINO/deployment_tools/inference_engine/lib/intel64/)

  find_library(
    OPENVINO_NGRAPH_LIBRARIES
    NAMES libngraph.so libngraph.a
    PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/deployment_tools/ngraph/lib $ENV{OPENVINO_ROOT}
          $ENV{OPENVINO_ROOT}/deployment_tools/ngraph/lib
          ${THIRD_PARTY_DIR}/OPENVINO/deployment_tools/ngraph/lib)

  find_library(
    OPENVINO_INFERENCE_LEGACY_LIBRARIES
    NAMES libinference_engine_legacy.so libinference_engine_legacy.a
    PATHS ${OPENVINO_ROOT} ${OPENVINO_ROOT}/deployment_tools/inference_engine/lib/intel64/
          $ENV{OPENVINO_ROOT} $ENV{OPENVINO_ROOT}/deployment_tools/inference_engine/lib/intel64/
          ${THIRD_PARTY_DIR}/OPENVINO/deployment_tools/inference_engine/lib/intel64/)

  list(APPEND OPENVINO_INCLUDE_DIR ${OPENVINO_INFERENCE_INCLUDE_DIR})
  list(APPEND OPENVINO_INCLUDE_DIR ${OPENVINO_NGRAPH_INCLUDE_DIR})
  list(APPEND OPENVINO_LIBRARIES ${OPENVINO_INFERENCE_LIBRARIES})
  list(APPEND OPENVINO_LIBRARIES ${OPENVINO_NGRAPH_LIBRARIES})
  list(APPEND OPENVINO_LIBRARIES ${OPENVINO_INFERENCE_LEGACY_LIBRARIES})

  if(OPENVINO_INCLUDE_DIR AND OPENVINO_LIBRARIES)

  else()
    message(
      FATAL_ERROR "OpenVINO was not found. You can set OPENVINO_ROOT to specify the search path.")
  endif()

  message(STATUS "OpenVINO Include: ${OPENVINO_INCLUDE_DIR}")
  message(STATUS "OpenVINO Lib: ${OPENVINO_LIBRARIES}")

  include_directories(${OPENVINO_INCLUDE_DIR})

endif(WITH_OPENVINO)
