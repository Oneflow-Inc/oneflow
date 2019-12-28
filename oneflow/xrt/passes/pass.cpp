#include "oneflow/xrt/passes/pass.h"

namespace oneflow {
namespace xrt {

bool CheckUseXrtEngine(const ClusteringOptions &options, const XrtEngine &engine) {
  XrtEngineOptionBit bit = [&]() {
    switch (engine) {
      case XrtEngine::XLA: return XrtEngineOptionBit::kUseXlaJit;
      case XrtEngine::TENSORRT: return XrtEngineOptionBit::kUseTensorRT;
      default: return XrtEngineOptionBit::kUseDefault;
    }
  }();

  return options.engine & (1U << bit);
}

}  // namespace xrt
}  // namespace oneflow
