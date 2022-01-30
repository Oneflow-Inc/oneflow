set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

oneflow_add_llvm_tool(oneflow-runner oneflow-runner.cpp)

target_link_libraries(
  oneflow-runner
  PRIVATE MLIRAnalysis
          MLIRExecutionEngine
          MLIRIR
          MLIRJitRunner
          MLIRLLVMIR
          MLIRLLVMToLLVMIRTranslation
          MLIRToLLVMIRTranslationRegistration
          MLIRParser
          MLIRTargetLLVMIRExport
          MLIRSupport
          MLIROneFlow)
