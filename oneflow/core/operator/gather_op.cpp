#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetSpecialConf() const {
  return op_conf().gather_conf();
}

void GatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *GetBlobDesc4BnInOp("in");
  out->set_max_col_num(1);
  out->set_has_col_num_field(false);
}

void GatherOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  if (kernel_conf->is_forward() == false) {
    kernel_conf->set_need_do_col_num(true);
  }
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
