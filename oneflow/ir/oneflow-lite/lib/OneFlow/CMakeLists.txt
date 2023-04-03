set(LITE_LOWERING_SRCS "")
set(LITE_LOWERING_LIBS "")

if(LITE_USE_ASCEND_NPU)
  include(cmake/FindAscendSdk.cmake)
  include_directories(${ASCEND_INCLUDE_DIR})
  include_directories(${ASCEND_INCLUDE_DIR}/../../opp)

  add_definitions(-DLITE_USE_ASCEND_NPU=1)

  list(APPEND LITE_LOWERING_SRCS Transform/Lowering/LoweringAscend.cpp)
  list(APPEND LITE_LOWERING_LIBS ${ASCEND_LIBRARIES})
endif()

oneflow_add_mlir_library(
  OneFlowLiteConversion
  ConvertToLiteExecutable.cpp
  FlatbufferUtils.cpp
  OneFlowLiteUtils.cpp
  Transform/FoldVariable.cpp
  Transform/InferPlacement.cpp
  Transform/InsertTransferOp.cpp
  Transform/MemoryPlanning.cpp
  Transform/PartitionLaunchJob.cpp
  Transform/LoweringLaunchJob.cpp
  ${LITE_LOWERING_SRCS}
  DEPENDS
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  MLIROneFlow
  MLIROneFlowExtension
  flatcc-runtime
  LINK_LIBS
  MLIRIR
  ${dialect_libs}
  ${translation_libs}
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  MLIROneFlow
  oneflow
  $<TARGET_OBJECTS:of_op_schema>
  MLIROneFlowExtension
  ${LITE_LOWERING_LIBS}
  $<BUILD_INTERFACE:flatcc-runtime>)
