#include "oneflow/core/operator/shared_model_diff_add_op.h"

namespace oneflow {

void SharedModelDiffAddOp::InitFromOpConf() {
  CHECK(op_conf().has_shared_model_diff_add_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().shared_model_diff_add_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("processed_model_diff", false);
  EnrollDataTmpBn("float_tmp");
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
  if (in_0_blob_desc->data_type() == DataType::kFloat16) {
    BlobDesc* dt_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
    *dt_blob_desc = *in_0_blob_desc;
    dt_blob_desc->set_data_type(DataType::kFloat);
  }
}

REGISTER_OP(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddOp);

}  // namespace oneflow
