#include "oneflow/core/kernel/kernel_registration.h"

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

Kernel* CreateKernel(const KernelConf& kernel_conf) {
  auto op_type = kernel_conf.op_attribute().op_conf().op_type_case();
  const auto& registry_vals = MutKernelRegistry()->at(op_type);

  Kernel* ret = nullptr;
  bool is_matched = false;
  for (const KernelRegistryVal& val : registry_vals) {
    if (val.cons->IsMatched(kernel_conf)) {
      CHECK(!is_matched)
          << "There are more than one kernel constraints satisfied by kernel conf of "
          << static_cast<size_t>(op_type);
      is_matched = true;
      ret = val.func();
    }
  }
  return ret;
}

}  // namespace kernel_registration

}  // namespace oneflow
