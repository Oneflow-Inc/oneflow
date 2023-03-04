add_mlir_pdll_library(MLIROneFlowPDLLForwardOpPatternsIncGen ForwardOpPatterns.pdll
                      ForwardOpPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLNormalizationPatternsIncGen NormalizationPatterns.pdll
                      NormalizationPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLFuseConv2DBatchNormPatternIncGen
                      FuseConv2DBatchNormPattern.pdll FuseConv2DBatchNormPattern.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLElementwiseFusionPatternsIncGen ElementwiseFusionPatterns.pdll
                      ElementwiseFusionPatterns.h.inc)
add_mlir_pdll_library(MLIROneFlowPDLLFuseOpsWithBackwardImplPatternsIncGen
                      FuseOpsWithBackwardImplPattern.pdll FuseOpsWithBackwardImplPattern.h.inc)
oneflow_add_mlir_dialect_library(
  MLIROneFlowPDLLPatterns
  ForwardOpPatterns.cpp
  NormalizationPatterns.cpp
  ElementwiseFusionPatterns.cpp
  FuseConv2DBatchNormPattern.cpp
  FuseOpsWithBackwardImplPattern.cpp
  DEPENDS
  MLIROneFlowPDLLForwardOpPatternsIncGen
  MLIROneFlowPDLLNormalizationPatternsIncGen
  MLIROneFlowPDLLElementwiseFusionPatternsIncGen
  MLIROneFlowPDLLFuseConv2DBatchNormPatternIncGen
  MLIROneFlowPDLLFuseOpsWithBackwardImplPatternsIncGen)
