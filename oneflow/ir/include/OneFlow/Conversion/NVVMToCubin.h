#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_NVVMTOCUBIN_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_NVVMTOCUBIN_H_

#ifdef WITH_MLIR_CUDA_CODEGEN

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace gpu {

namespace ptx {

inline std::string getPTXTriple() { return "gpu.triple"; }
inline std::string getPTXChip() { return "gpu.chip"; }
inline std::string getPTXFeatures() { return "gpu.features"; }

}  // namespace ptx

inline std::string getCubinAnnotation() { return "gpu.binary"; }

}  // namespace gpu

void InitializeLLVMNVPTXBackend();
std::unique_ptr<mlir::Pass> createNVVMToCubinPass();

}  // namespace mlir

#endif  // WITH_MLIR_CUDA_CODEGEN

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_NVVMTOCUBIN_H_