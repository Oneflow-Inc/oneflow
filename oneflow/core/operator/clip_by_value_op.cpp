#include "oneflow/core/operator/clip_by_value_op.h"

namespace oneflow {

void ClipByValueOp::InitFromOpConf() {
  CHECK(op_conf().has_clip_by_value_conf());
  EnrollInputBn("in");
  EnrollDataTmpBn("clip_mask");
  EnrollOutputBn("out");
}

const PbMessage& ClipByValueOp::GetCustomizedConf() const {
  return this->op_conf().clip_by_value_conf();
}

void ClipByValueOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  // data_tmp: clip_mask
  BlobDesc* clip_mask = GetBlobDesc4BnInOp("clip_mask");
  *clip_mask = *in;
  clip_mask->set_data_type(kInt8);
  // output
  *GetBlobDesc4BnInOp("out") = *in;
}

REGISTER_OP(OperatorConf::kClipByValueConf, ClipByValueOp);

}  // namespace oneflow
