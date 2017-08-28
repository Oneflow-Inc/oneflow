#include "oneflow/core/operator/relu_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReluOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReluOp::GetSpecialConf() const {
  return op_conf().relu_conf();
}

void ReluOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const ReluOpConf& conf = op_conf().relu_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(), conf.in().data_type());
  CHECK_EQ(in_blob_desc->data_type(), conf.out().data_type());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = in_blob_desc->shape();
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

}  // namespace oneflow
