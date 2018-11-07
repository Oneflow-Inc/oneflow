#include "oneflow/core/operator/prelu_op.h"

namespace oneflow {

void PReluOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_conf());
  EnrollInputBn("in");
  EnrollModelBn("weight");
  EnrollOutputBn("out");
}

const PbMessage& PReluOp::GetCustomizedConf() const { return op_conf().prelu_conf(); }

void PReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const oneflow::PReluOpConf conf = op_conf().prelu_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  if (!conf.channel_shared()) {
    if (conf.data_format() == "channels_first") {
      weight_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(1)});
    } else if (conf.data_format() == "channels_last") {
      weight_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(3)});
    } else {
      UNIMPLEMENTED();
    }
  } else {
    weight_blob_desc->mut_shape() = Shape({1});
  }
  weight_blob_desc->set_data_type(in_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kPreluConf, PReluOp);

}  // namespace oneflow
