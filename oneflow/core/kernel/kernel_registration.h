#ifndef ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"

namespace std {
template<>
struct hash<::oneflow::OperatorConf::OpTypeCase> {
  std::size_t operator()(const ::oneflow::OperatorConf::OpTypeCase& op_type) const {
    return static_cast<size_t>(op_type);
  }
};
}  // namespace std

namespace oneflow {

namespace kernel_registration {

class Kernel;
class KernelRegistryVal;

using KernelRegMap = HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>>;
using CreateFn = std::function<Kernel*()>;

namespace constraint {

class KernelConstraint {
 public:
  KernelConstraint() = default;
  virtual ~KernelConstraint() = default;

  virtual bool IsMatched(const KernelConf&) = 0;
  virtual size_t PriorityLevel() { return 0; }  // big number means high priority
};

class DeviceAndDTypeConstraint final : public KernelConstraint {
 public:
  DeviceAndDTypeConstraint(DeviceType dev, DataType dtype) : dev_(dev), dtype_(dtype) {}
  bool IsMatched(const KernelConf&) override;

 private:
  DeviceType dev_;
  DataType dtype_;
};

class DeviceConstraint final : public KernelConstraint {
 public:
  DeviceConstraint(DeviceType dev) : dev_(dev) {}
  bool IsMatched(const KernelConf&) override;

 private:
  DeviceType dev_;
};

}  // namespace constraint

struct KernelRegistryVal {
  CreateFn func;
  std::shared_ptr<constraint::KernelConstraint> cons;

  KernelRegistryVal(CreateFn f, const std::shared_ptr<constraint::KernelConstraint>& c)
      : func(f), cons(c) {}
};

struct KernelRegistrar final {
  KernelRegistrar(const OperatorConf::OpTypeCase& op_type, constraint::KernelConstraint* cons,
                  CreateFn f);
};

}  // namespace kernel_registration

#define KERNEL_REGISTER_WITH_DEVICE_AND_DTYPE(op_type, device, dtype, ...)                        \
  namespace {                                                                                     \
  static kernel_registration::KernelRegistrar(                                                    \
      op_type,                                                                                    \
      kernel_registration::constraint::DeviceAndDTypeConstraint(device, GetDataType<dtype>::val), \
      []() { return __VA_ARGS__; });                                                              \
  }  // namespace

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
