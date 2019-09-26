#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace kernel_registration {

namespace {

struct KernelRegistryVal final {
  KernelRegistryVal(CreateFn f, const std::shared_ptr<constraint::KernelConstraint>& c)
      : func(f), cons(c) {}

  CreateFn func;
  std::shared_ptr<constraint::KernelConstraint> cons;
};

HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>>* MutKernelRegistry() {
  static HashMap<OperatorConf::OpTypeCase, std::vector<KernelRegistryVal>> creators;
  return &creators;
}

}  // namespace

namespace constraint {

void NoConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  *(val->mutable_device_and_dtypes()->Add()) = KernelRegValProto::Device7DType();
}

bool DeviceAndDTypeConstraint::IsMatched(const KernelConf& kernel_conf) const {
  return (dev_ == kernel_conf.op_attribute().op_conf().device_type()
          && dtype_ == kernel_conf.data_type());
}

void DeviceAndDTypeConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  KernelRegValProto::Device7DType proto;
  proto.set_device(dev_);
  proto.add_dtype(dtype_);
  *(val->mutable_device_and_dtypes()->Add()) = proto;
}

bool DeviceConstraint::IsMatched(const KernelConf& kernel_conf) const {
  return dev_ == kernel_conf.op_attribute().op_conf().device_type();
}

void DeviceConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  KernelRegValProto::Device7DType proto;
  proto.set_device(dev_);
  *(val->mutable_device_and_dtypes()->Add()) = proto;
}

bool PredAndLabelConstraint::IsMatched(const KernelConf& kernel_conf) const {
  DataType pred_type;
  DataType label_type;
  LossKernelConf loss_conf;
  if (kernel_conf.kernel_type_case() == KernelConf::kSparseSoftmaxCrossEntropyLossConf) {
    pred_type = kernel_conf.sparse_softmax_cross_entropy_loss_conf().loss_conf().prediction_type();
    label_type = kernel_conf.sparse_softmax_cross_entropy_loss_conf().loss_conf().label_type();
  } else if (kernel_conf.kernel_type_case() == KernelConf::kAccuracyConf) {
    pred_type = kernel_conf.accuracy_conf().prediction_type();
    label_type = kernel_conf.accuracy_conf().label_type();
  } else {
    UNIMPLEMENTED();
  }
  return (pred_type == pred_type_ && label_type == label_type_);
}

void PredAndLabelConstraint::ToProto(KernelRegValProto::RegVal* val) const {
  KernelRegValProto::Device7DType proto;
  proto.set_device(dev_);
  proto.add_dtype(pred_type_);
  proto.add_dtype(label_type_);
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
  auto kernel_registry = MutKernelRegistry();
  if (kernel_registry->find(op_type) == kernel_registry->end()) { return nullptr; }
  const auto& registry_vals = kernel_registry->at(op_type);

  Kernel* ret = nullptr;
  bool is_matched = false;
  for (const KernelRegistryVal& val : registry_vals) {
    if (val.cons->IsMatched(kernel_conf)) {
      CHECK(!is_matched)
          << "There are more than one kernel constraints satisfied by kernel conf of "
          << static_cast<size_t>(op_type);
      is_matched = true;
      ret = val.func();
      ret->set_device_type(val.cons->device_type());
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
