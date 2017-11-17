#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf() {
  CHECK(op_conf().has_softmax_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");

#ifndef USE_CUDNN
  EnrollDataTmpBn("tmp");
#endif  // USE_CUDNN
}

const PbMessage& SoftmaxOp::GetSpecialConf() const {
  return op_conf().softmax_conf();
}

void SoftmaxOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  // in
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  // out
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({in->shape().At(0), in->shape().Count(1)});
  out->set_data_type(in->data_type());
  out->set_has_data_id(in->has_data_id());
  CHECK_EQ(in->data_type(), out->data_type());

#ifndef USE_CUDNN
  // tmp
  BlobDesc* tmp = GetBlobDesc4BnInOp("tmp");
  tmp->mut_shape() = Shape({in->shape().At(0)});
  tmp->set_data_type(in->data_type());
  tmp->set_has_data_id(false);
#endif  // USE_CUDNN
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
