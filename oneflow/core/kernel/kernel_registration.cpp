#include "oneflow/core/kernel/kernel_registration.h"

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

namespace {

struct KernelRegistryVal final {
  CreateFn func;
  std::shared_ptr<constraint::KernelConstraint> cons;

  KernelRegistryVal(CreateFn f, const std::shared_ptr<constraint::KernelConstraint>& c)
      : func(f), cons(c) {}
};

using KernelRegMap = HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>>;

KernelRegMap* MutKernelRegistry() {
  static KernelRegMap creators;
  return &creators;
}

}  // namespace

namespace constraint {

bool DeviceAndDTypeConstraint::IsMatched(const KernelConf& kernel_conf) {
  return (dev_ == kernel_conf.op_attribute().op_conf().device_type()
          && dtype_ == kernel_conf.data_type());
}

bool DeviceConstraint::IsMatched(const KernelConf& kernel_conf) {
  return dev_ == kernel_conf.op_attribute().op_conf().device_type();
}

}  // namespace constraint

KernelRegistrar::KernelRegistrar(const OperatorConf::OpTypeCase& op_type,
                                 constraint::KernelConstraint* cons, CreateFn f) {
  auto* creators = MutKernelRegistry();
  (*creators)[op_type].emplace_back(f, std::shared_ptr<constraint::KernelConstraint>(cons));
}

}  // namespace kernel_registration

}  // namespace oneflow
