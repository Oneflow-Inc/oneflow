oneflow_add_mlir_library(
  MLIROneFlowExtension
  extension.cpp
  ir_pass.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRIR
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslateLib
  MLIRSupport
  MLIROneFlow
  oneflow
  MLIRExecutionEngine
  MLIROneFlowTranslation
  MLIROneFlowRuntime
  ENABLE_AGGREGATION)
set(MLIR_ONEFLOW_JIT_LIBS -Wl,--no-as-needed MLIROneFlowJIT -Wl,--as-needed)
target_link_libraries(MLIROneFlowRuntime PUBLIC ${MLIR_ONEFLOW_JIT_LIBS})
mlir_check_all_link_libraries(MLIROneFlowExtension)
add_custom_target(mex DEPENDS MLIROneFlowExtension)
