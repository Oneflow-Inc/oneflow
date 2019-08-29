#include "oneflow/core/operator/basic_rnn_op.h"

namespace oneflow {

const PbMessage& BasicRnnOp::GetCustomizedConf() const { return op_conf().basic_rnn_conf(); }

void BasicRnnOp::VirtualInitFromOpConf() {
  EnrollTmpBn("plus_op_out");
  EnrollTmpBn("i2h_weight");
  EnrollTmpBn("h2h_weight");
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    EnrollTmpBn("bias");
    EnrollConstBufBn("bias_multiplier");
  }
}

void BasicRnnOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  // int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);
  *GetBlobDesc4BnInOp("plus_op_out") =
      BlobDesc(Shape({data_num, hidden_size}), GlobalJobDesc().DefaultDataType(), false, true,
               in_blob_desc->max_col_num());
  // *GetBlobDesc4BnInOp("i2h_weight") = BlobDesc(Shape({hidden_size, embedding_size}));
  // *GetBlobDesc4BnInOp("h2h_weight") = BlobDesc(Shape({hidden_size, hidden_size}));
  TODO();
  if (GetValFromCustomizedConf<bool>("use_bias")) {
    // *GetBlobDesc4BnInOp("bias") = BlobDesc(Shape({1, hidden_size}));
    // *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({data_num, 1}));
    TODO();
  }
}

REGISTER_OP(OperatorConf::kBasicRnnConf, BasicRnnOp);

}  // namespace oneflow
