#include "oneflow/core/operator/process_model_diff_op.h"

namespace oneflow {

void ProcessModelDiffOp::InitFromOpConf() {
  CHECK(op_conf().has_process_model_diff_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().process_model_diff_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("processed_model_diff", false);
}
const PbMessage& ProcessModelDiffOp::GetCustomizedConf() const {
  return op_conf().process_model_diff_conf();
}

void ProcessModelDiffOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  FOR_RANGE(int32_t, i, 1, input_bns().size()) {
    CHECK(*in_0_blob_desc == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
}

REGISTER_OP(OperatorConf::kProcessModelDiffConf, ProcessModelDiffOp);

}  // namespace oneflow
