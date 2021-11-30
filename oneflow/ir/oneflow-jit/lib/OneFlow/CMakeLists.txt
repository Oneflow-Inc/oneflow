add_llvm_library(MLIROneFlowJIT
  JIT.cpp
  jit_op_interpreter.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/oneflow-translate/include
  ${PROJECT_SOURCE_DIR}/oneflow-jit/include
  DEPENDS oneflow_deps
	LINK_LIBS PUBLIC
	MLIRIR
  BUILDTREE_ONLY
)

llvm_update_compile_flags(MLIROneFlowJIT)

target_link_libraries(MLIROneFlowJIT
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIROneFlow
  MLIRExecutionEngine
  MLIROneFlowTranslation
)

mlir_check_link_libraries(MLIROneFlowJIT)
