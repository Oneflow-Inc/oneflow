#include "oneflow/core/kernel/kernel_registration.h"

namespace oneflow {

namespace kernel_registration {

namespace {

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
