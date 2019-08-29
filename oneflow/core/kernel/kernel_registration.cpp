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

HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>>* MutKernelRegistry() {
  static HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>> creators;
  return &creators;
}

}  // namespace

namespace constraint {

bool DeviceAndDTypeConstraint::IsMatched(const KernelConf& kernel_conf) {
  return (dev_ == kernel_conf.op_attribute().op_conf().device_type()
          && dtype_ == kernel_conf.data_type());
}

void DeviceAndDTypeConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  KernelRegValProto::Device7DType proto;
  proto.set_device(dev_);
  proto.add_dtype(dtype_);
  *(val->mutable_device_and_dtypes()->Add()) = proto;
}

bool DeviceConstraint::IsMatched(const KernelConf& kernel_conf) {
  return dev_ == kernel_conf.op_attribute().op_conf().device_type();
}

void DeviceConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  KernelRegValProto::Device7DType proto;
  proto.set_device(dev_);
  *(val->mutable_device_and_dtypes()->Add()) = proto;
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

void ExportProtoFromKernelRegistry(KernelRegValProto* proto) {
  auto* creators = MutKernelRegistry();
  for (const auto& pair : *creators) {
    KernelRegValProto::RegVal reg_val_proto;
    for (const auto& registry_val : pair.second) { registry_val.cons->ToProto(&reg_val_proto); }
    proto->mutable_kernel2reg_val()->insert({static_cast<int64_t>(pair.first), reg_val_proto});
  }
}

}  // namespace kernel_registration

}  // namespace oneflow
