#ifdef WITH_MLIR_CUDA_CODEGEN
#include "oneflow/core/common/util.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

#include <cuda.h>

static void emitCudaError(const llvm::Twine& expr, const char* buffer, CUresult result,
                          mlir::Location loc) {
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

namespace mlir {
namespace {

const std::string& getLibDevice() {
  static std::string p;
  if (p.size() > 0) return p;
  const auto toolkit_env_name = "CUDA_TOOLKIT_ROOT_DIR";
  p = ::oneflow::GetStringFromEnv(toolkit_env_name, "/usr/local/cuda/")
      + "nvvm/libdevice/libdevice.10.bc";
  return p;
}

LogicalResult linkLibdevice(llvm::Module& llvmModule, llvm::LLVMContext& llvmContext) {
  // Note: infer libdevice path from environment variable
  auto libDevice = getLibDevice();

  // Note: load raw data from file
  std::string errorMessage;
  auto libDeviceBuf = openInputFile(libDevice, &errorMessage);
  if (!libDeviceBuf) LOG(FATAL) << "Open File error when link libdevice: " << errorMessage;

  // Note: load module from raw data
  auto moduleOrErr = llvm::getOwningLazyBitcodeModule(std::move(libDeviceBuf), llvmContext);
  if (!moduleOrErr) LOG(FATAL) << "Failed to load: " << libDevice << "\n";
  std::unique_ptr<llvm::Module> libDeviceModule = std::move(moduleOrErr.get());

  // Note: link libdevice with module
  if (llvm::Linker::linkModules(llvmModule, std::move(libDeviceModule),
                                llvm::Linker::Flags::LinkOnlyNeeded,
                                [](llvm::Module& M, const llvm::StringSet<>& GS) {
                                  llvm::internalizeModule(M, [&GS](const llvm::GlobalValue& GV) {
                                    return !GV.hasName() || (GS.count(GV.getName()) == 0);
                                  });
                                })) {
    LOG(FATAL) << "failed to link libdevice module\n";
  }

  return success();
}

class NVVMToCubinPass : public NVVMToCubinPassBase<NVVMToCubinPass> {
  std::unique_ptr<llvm::Module> translateToLLVMIR(llvm::LLVMContext& llvmContext) {
    return translateModuleToLLVMIR(getOperation(), llvmContext, "LLVMDialectModule");
  }

 public:
  std::optional<std::string> translateToISA(llvm::Module& llvmModule,
                                            llvm::TargetMachine& targetMachine) {
    llvmModule.setDataLayout(targetMachine.createDataLayout());

    // TODO: optimizeLlvm

    std::string targetISA;
    llvm::raw_string_ostream stream(targetISA);

    {  // Drop pstream after this to prevent the ISA from being stuck buffering
      llvm::buffer_ostream pstream(stream);
      llvm::legacy::PassManager codegenPasses;

      if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                            llvm::CGFT_AssemblyFile))
        return std::nullopt;

      codegenPasses.run(llvmModule);
    }
    return stream.str();
  }
  std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
    Location loc = getOperation().getLoc();
    std::string error;
    const llvm::Target* target = ::llvm::TargetRegistry::lookupTarget(triple.str(), error);
    if (!target) {
      emitError(loc, Twine("failed to lookup target: ") + error);
      return {};
    }
    llvm::TargetMachine* machine =
        target->createTargetMachine(triple.str(), chip.str(), features.str(), {}, {});
    if (!machine) {
      emitError(loc, "failed to create target machine");
      return {};
    }

    return std::unique_ptr<llvm::TargetMachine>{machine};
  }
  std::unique_ptr<std::vector<char>> serializeISA(const std::string& isa) {
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

  void runOnOperation() override {
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);
    if (!llvmModule) return signalPassFailure();
    if (failed(linkLibdevice(*llvmModule, llvmContext))) { return signalPassFailure(); }

    // Lower the LLVM IR module to target ISA.
    std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine();
    if (!targetMachine) return signalPassFailure();

    std::optional<std::string> maybeTargetISA = translateToISA(*llvmModule, *targetMachine);

    if (!maybeTargetISA.has_value()) return signalPassFailure();

    std::string targetISA = std::move(*maybeTargetISA);

    // Serialize the target ISA.
    std::unique_ptr<std::vector<char>> blob = serializeISA(targetISA);
    if (!blob) return signalPassFailure();

    // Add the blob as module attribute.
    auto attr = StringAttr::get(&getContext(), StringRef(blob->data(), blob->size()));
    getOperation()->setAttr(gpu::getCubinAnnotation(), attr);
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registerLLVMDialectTranslation(registry);
    registerNVVMDialectTranslation(registry);
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> createNVVMToCubinPass() { return std::make_unique<NVVMToCubinPass>(); }

void InitializeLLVMNVPTXBackend() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

}  // namespace mlir
#endif  // WITH_MLIR_CUDA_CODEGEN