#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowDevice.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oneflow {

namespace {
class OneFlowRequestDeviceWrappersPass
    : public OneFlowRequestDeviceWrappersBase<OneFlowRequestDeviceWrappersPass> {
 public:
  void runOnOperation() override final {
    auto name =
        device::DeviceBuilder().device(deviceType).done()->getWrapperName();
    getOperation()->setAttr(name, UnitAttr::get(&getContext()));
  }
};
}  // namespace

std::unique_ptr<Pass> createOneFlowRequestDeviceWrappers() {
  return std::make_unique<OneFlowRequestDeviceWrappersPass>();
}

}  // namespace oneflow
}  // namespace mlir
