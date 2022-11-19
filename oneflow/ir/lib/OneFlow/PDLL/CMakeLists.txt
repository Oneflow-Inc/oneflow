add_mlir_pdll_library(MLIROneFlowPDLLPatternsIncGen
  OneFlowPatterns.pdll
  OneFlowPatterns.h.inc
)
message(STATUS "MLIROneFlowPDLLPatternsIncGen:   ${PROJECT_BINARY_DIR}")
oneflow_add_mlir_dialect_library(MLIROneFlowPDLLPatterns
  OneFlowPDLLPatterns.cpp

  DEPENDS
  MLIROneFlowPDLLPatternsIncGen
)
