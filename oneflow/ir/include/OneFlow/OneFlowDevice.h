#ifndef ONEFLOW_ONEFLOW_DEVICE_H_
#define ONEFLOW_ONEFLOW_DEVICE_H_

#include "oneflow/core/common/util.h"
#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/OneFlowOpsDialect.h.inc"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <memory>
#include <utility>
#include <glog/logging.h>

namespace mlir {

namespace oneflow {

// Note: this namespace may be outline to a single dialect in the future
namespace device {

class DeviceProto {
  std::string getNamespaceStr() const { return OneFlowDialect::getDialectNamespace().str(); }

 protected:
  std::string version_;

 public:
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

class DeviceBuilder final {
  static DeviceBuilder instance_;
  std::unique_ptr<DeviceProto> proto_;

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

}  // namespace device

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_ONEFLOW_DEVICE_H_
