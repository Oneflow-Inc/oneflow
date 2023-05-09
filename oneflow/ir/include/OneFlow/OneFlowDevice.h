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
#ifndef ONEFLOW_ONEFLOW_DEVICE_H_
#define ONEFLOW_ONEFLOW_DEVICE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "oneflow/core/common/util.h"
#include "OneFlow/OneFlowDialect.h"

#include <memory>
#include <utility>
#include <glog/logging.h>

namespace mlir {

namespace oneflow {

// Note: this namespace may be outline to a single dialect in the future
namespace device {

class DeviceProto {
 protected:
  std::string version_;

 public:
  static std::string getNamespaceStr() { return OneFlowDialect::getDialectNamespace().str(); }
  virtual ~DeviceProto() = default;

  virtual std::string getName() const = 0;
  virtual void setVersion(const std::string& version) = 0;
  std::string getVersion() { return version_; }
  const std::string getWrapperName() const { return getNamespaceStr() + "." + getName(); }
};

class GPUDevice final : public DeviceProto {
 public:
  inline const static std::string TAG = "gpu";
  std::string getName() const override { return TAG; }
  void setVersion(const std::string& version) override { TODO(); }
};

class CPUDevice final : public DeviceProto {
 public:
  inline const static std::string TAG = "cpu";
  std::string getName() const override { return TAG; }
  void setVersion(const std::string& version) override { version_ = "sm_" + version; }
};

class DeviceBuilder {
  std::unique_ptr<DeviceProto> proto_;

 protected:
  static std::unique_ptr<DeviceProto> fromName(const std::string& name) {
    if (name == GPUDevice::TAG) return std::make_unique<GPUDevice>();
    if (name == CPUDevice::TAG) return std::make_unique<CPUDevice>();

    LOG(FATAL) << "Fail to build device proto from name: " << name;
  }

 public:
  DeviceBuilder& device(const std::string& name) {
    proto_ = fromName(name);
    return *this;
  }

  DeviceBuilder& version(const std::string& version) {
    proto_->setVersion(version);
    return *this;
  }

  std::unique_ptr<DeviceProto> done() {
    std::unique_ptr<DeviceProto> ret;
    ret.swap(proto_);
    return ret;
  }
};

class FuncOp2DeviceBuilder final : private DeviceBuilder {
  func::FuncOp func_;

 public:
  explicit FuncOp2DeviceBuilder(func::FuncOp func) : func_(func) {
    DeviceBuilder::device(isGPU() ? "gpu" : "cpu");
  }

  bool isGPU() {
    return func_->getAttr(DeviceProto::getNamespaceStr() + "." + GPUDevice::TAG) != nullptr;
  }
  bool isCPU() {
    return func_->getAttr(DeviceProto::getNamespaceStr() + "." + CPUDevice::TAG) != nullptr;
  }
};
}  // namespace device

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_ONEFLOW_DEVICE_H_
