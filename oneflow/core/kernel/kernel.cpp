#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::InitFromOpProto(const OperatorProto& op_proto) {
  Operator* op = CreateOp(op_proto.op_conf().op_type_case());
  op->InitFromProto(op_proto);
  op_.reset(op);
}

void Kernel::InitModelAndModelTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> Blob4BnInOp) const {
  TODO();
}

} // namespace oneflow
