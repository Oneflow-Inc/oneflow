/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*
This file is ported from mlir/lib/Dialect/GPU/Transforms/SerializeToCubin.cpp
*/

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#ifdef WITH_MLIR_CUDA_CODEGEN

#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace mlir;

static void emitCudaError(const llvm::Twine& expr, const char* buffer, CUresult result,
                          Location loc) {
  const char* error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                       \
  do {                                                   \
    if (auto status = (expr)) {                          \
      emitCudaError(#expr, jitErrorBuffer, status, loc); \
      return {};                                         \
    }                                                    \
  } while (false)

namespace {
class SerializeToCubinPass : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)
  SerializeToCubinPass();

  StringRef getArgument() const override { return "out-of-tree-gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override;

  // Serializes PTX to CUBIN.
  std::unique_ptr<std::vector<char>> serializeISA(const std::string& isa) override;
};
}  // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string>& option, const char* value) {
  if (!option.hasValue()) option = value;
}

SerializeToCubinPass::SerializeToCubinPass() {
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
    exit(1);
  }
  std::string arch = std::to_string(prop.major) + std::to_string(prop.minor);
  maybeSetOption(this->triple, "nvptx64-nvidia-cuda");
  maybeSetOption(this->chip, ("sm_" + arch).c_str());
  std::string ptx_arch = arch;
  // NOTE: doesn't support PTX75 for now
  if (ptx_arch == "75") { ptx_arch = "72"; }
  maybeSetOption(this->features, ("+ptx" + ptx_arch).c_str());
}

void SerializeToCubinPass::getDependentDialects(DialectRegistry& registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<std::vector<char>> SerializeToCubinPass::serializeISA(const std::string& isa) {
  Location loc = getOperation().getLoc();
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* jitOptionsVals[] = {jitErrorBuffer, reinterpret_cast<void*>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState));

  auto kernelName = getOperation().getName().str();
  RETURN_ON_CUDA_ERROR(cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                                     const_cast<void*>(static_cast<const void*>(isa.c_str())),
                                     isa.length(), kernelName.c_str(),
                                     0,       /* number of jit options */
                                     nullptr, /* jit options */
                                     nullptr  /* jit option values */
                                     ));

  void* cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize));

  char* cubinAsChar = static_cast<char*>(cubinData);
  auto result = std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context));

  return result;
}

namespace mlir {

namespace oneflow {

void InitializeLLVMNVPTXBackend() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

void registerGpuSerializeToCubinPass() {
  PassRegistration<SerializeToCubinPass> registerSerializeToCubin([] {
    InitializeLLVMNVPTXBackend();
    return std::make_unique<SerializeToCubinPass>();
  });
}

std::unique_ptr<Pass> createSerializeToCubinPass() {
  return std::make_unique<SerializeToCubinPass>();
}

}  // namespace oneflow

}  // namespace mlir

#endif  // WITH_MLIR_CUDA_CODEGEN
