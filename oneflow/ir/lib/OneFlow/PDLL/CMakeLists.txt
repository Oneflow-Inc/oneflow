add_mlir_pdll_library(MLIROneFlowPDLLForwardOpPatternsIncGen ForwardOpPatterns.pdll ForwardOpPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLNormalizationPatternsIncGen NormalizationPatterns.pdll NormalizationPatterns.h.inc)
oneflow_add_mlir_dialect_library(MLIROneFlowPDLLPatterns ForwardOpPatterns.cpp NormalizationPatterns.cpp DEPENDS
                                MLIROneFlowPDLLForwardOpPatternsIncGen MLIROneFlowPDLLNormalizationPatternsIncGen)
