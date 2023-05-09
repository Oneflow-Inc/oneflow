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
    auto name = device::DeviceBuilder().device(deviceType).done()->getWrapperName();
    getOperation()->setAttr(name, UnitAttr::get(&getContext()));
  }
};
}  // namespace

std::unique_ptr<Pass> createOneFlowRequestDeviceWrappers() {
  return std::make_unique<OneFlowRequestDeviceWrappersPass>();
}

}  // namespace oneflow
}  // namespace mlir
