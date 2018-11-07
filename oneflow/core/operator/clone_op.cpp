#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

void CloneOp::InitFromOpConf() {
  EnrollInputBn("in");
  for (int64_t i = 0; i < op_conf().clone_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i));
  }
}

const PbMessage& CloneOp::GetCustomizedConf() const { return op_conf().clone_conf(); }

void CloneOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const BlobDesc* input_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  for (std::string obn : output_bns()) { *GetBlobDesc4BnInOp(obn) = *input_blob_desc; }
}

void CloneOp::InferDiffBlobDescsWithoutFwBlob(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp(output_diff_bns().Get(0));
  *GetBlobDesc4BnInOp(SoleIdbn()) = *out_diff_blob_desc;
}

void CloneOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  EraseEmptyBnInVec(GetBlobDesc4BnInOp,
                    kernel_conf->mutable_op_attribute()->mutable_output_diff_bns());
  if (!kernel_conf->is_forward()) {
    for (const std::string& output_diff_bn : output_diff_bns()) {
      const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp(output_diff_bn);
      if (out_diff_blob_desc == nullptr) { continue; }
      if (out_diff_blob_desc->has_dim0_valid_num_field()) {
        kernel_conf->set_need_do_dim0_valid_num(true);
        kernel_conf->set_can_naive_do_dim0_valid_num(true);
      }
      break;
    }
  }
}

REGISTER_OP(OperatorConf::kCloneConf, CloneOp);

}  // namespace oneflow
