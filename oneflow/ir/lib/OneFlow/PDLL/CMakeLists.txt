add_mlir_pdll_library(MLIROneFlowPDLLAllocEliminaionPatternsIncGen AllocEliminationPatterns.pdll
                      AllocEliminationPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLForwardOpPatternsIncGen ForwardOpPatterns.pdll
                      ForwardOpPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLNormalizationPatternsIncGen NormalizationPatterns.pdll
                      NormalizationPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLFuseConv2DBatchNormPatternIncGen
                      FuseConv2DBatchNormPattern.pdll FuseConv2DBatchNormPattern.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLFuseOpsWithBackwardImplPatternsIncGen
                      FuseOpsWithBackwardImplPattern.pdll FuseOpsWithBackwardImplPattern.h.inc)
oneflow_add_mlir_dialect_library(
  MLIROneFlowPDLLPatterns
  AllocEliminationPatterns.cpp
  ForwardOpPatterns.cpp
  NormalizationPatterns.cpp
  FuseConv2DBatchNormPattern.cpp
  FuseOpsWithBackwardImplPattern.cpp
  DEPENDS
  MLIROneFlowPDLLAllocEliminaionPatternsIncGen
  MLIROneFlowPDLLForwardOpPatternsIncGen
  MLIROneFlowPDLLNormalizationPatternsIncGen
  MLIROneFlowPDLLFuseConv2DBatchNormPatternIncGen
  MLIROneFlowPDLLFuseOpsWithBackwardImplPatternsIncGen)
