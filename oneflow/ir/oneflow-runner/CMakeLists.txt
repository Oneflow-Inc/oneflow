set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

oneflow_add_llvm_tool(oneflow-runner oneflow-runner.cpp)

set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()
set_target_properties(
  oneflow-runner PROPERTIES BUILD_WITH_INSTALL_RPATH OFF BUILD_RPATH "${_origin_prefix}"
                            INSTALL_RPATH "${_origin_prefix}")

target_link_libraries(
  oneflow-runner
  PRIVATE MLIRAnalysis
          MLIRExecutionEngine
          MLIRIR
          MLIRJitRunner
          MLIRLLVMIRTransforms
          MLIRLLVMToLLVMIRTranslation
          MLIRToLLVMIRTranslationRegistration
          MLIRParser
          MLIRTargetLLVMIRExport
          MLIRSupport
          MLIROneFlow
          glog::glog)
