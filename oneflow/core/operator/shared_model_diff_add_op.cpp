#include "oneflow/core/operator/shared_model_diff_add_op.h"

namespace oneflow {

void SharedModelDiffAddOp::InitFromOpConf() {
  CHECK(op_conf().has_shared_model_diff_add_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().shared_model_diff_add_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("out");
}
const PbMessage& SharedModelDiffAddOp::GetCustomizedConf() const {
  return op_conf().shared_model_diff_add_conf();
}

void SharedModelDiffAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  FOR_RANGE(int32_t, i, 1, input_bns().size()) {
    CHECK(*in_0_blob_desc == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
  *GetBlobDesc4BnInOp("out") = *in_0_blob_desc;
}

REGISTER_OP(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddOp);

}  // namespace oneflow
