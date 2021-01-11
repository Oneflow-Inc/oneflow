set(LLVM_TARGET_DEFINITIONS OneFlowTranslateRewrites.td)
mlir_tablegen(OneFlowTranslateRewrites.inc -gen-rewriters -I${PROJECT_SOURCE_DIR}/include/OneFlow)
add_public_tablegen_target(OneFlowTranslateRewritesIncGen)
