#include "oneflow/core/operator/model_save_op.h"

namespace oneflow {

void ModelSaveOp::InitFromOpConf() {
  CHECK(op_conf().has_model_save_conf());
  for (const std::string& lbn : op_conf().model_save_conf().lbns()) {
    EnrollInputBn("in_" + lbn);
  }
}

const PbMessage& ModelSaveOp::GetSpecialConf() const {
  return op_conf().model_save_conf();
}

void ModelSaveOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    bool is_forward, const ParallelContext* parallel_ctx,
    KernelConf* kernel_conf) const {
  int64_t parallel_id = parallel_ctx->parallel_id();
  int64_t parallel_num = parallel_ctx->parallel_num();
  ParallelPolicy policy = parallel_ctx->policy();
  int32_t part_id = -1;
  int32_t total_part_num = -1;
  if (policy == kDataParallel) {
    part_id = 0;
    total_part_num = 1;
    CHECK_EQ(parallel_id, 0);
  } else if (policy == kModelParallel) {
    part_id = parallel_id;
    total_part_num = parallel_num;
  } else {
    UNEXPECTED_RUN();
  }
  ModelSaveKernelConf* model_save_kernel_conf =
      kernel_conf->mutable_model_save_conf();
  model_save_kernel_conf->set_part_id(part_id);
  model_save_kernel_conf->set_total_part_num(total_part_num);
}

REGISTER_OP(OperatorConf::kModelSaveConf, ModelSaveOp);

}  // namespace oneflow
