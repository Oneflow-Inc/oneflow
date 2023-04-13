include_directories(${PROJECT_SOURCE_DIR}/oneflow-extension/include)

oneflow_add_mlir_library(
  MLIROneFlowExtension
  mlir_jit_op.cpp
  mlir_jit_op_kernel.cpp
  ir_pass.cpp
  lr_jit.cpp
  mlir_gen.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRIR
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  MLIROneFlow
  oneflow
  MLIRExecutionEngine
  MLIROneFlowTranslation
  MLIROneFlowRuntime)
mlir_check_all_link_libraries(MLIROneFlowExtension)
add_custom_target(mex DEPENDS MLIROneFlowExtension)
