add_mlir_pdll_library(MLIROneFlowPDLLPatternsIncGen
  ForwardOpPatterns.pdll
  ForwardOpPatterns.h.inc
)
message(STATUS "MLIROneFlowPDLLPatternsIncGen:   ${PROJECT_BINARY_DIR}")
oneflow_add_mlir_dialect_library(MLIROneFlowPDLLPatterns
  ForwardOpPatterns.cpp
  DEPENDS
  MLIROneFlowPDLLPatternsIncGen
)
