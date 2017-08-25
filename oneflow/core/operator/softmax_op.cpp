#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

void SoftmaxOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("tmp");
}

const PbMessage& SoftmaxOp::GetSpecialConf() const {
  return op_conf().softmax_conf();
}

void SoftmaxOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const SoftmaxOpConf& conf = op_conf().softmax_conf();
  // CHECK data type
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(), conf.in().data_type());
  CHECK_EQ(conf.in().data_type(), JobDesc::Singleton()->default_data_type());
  CHECK_EQ(conf.out().data_type(), JobDesc::Singleton()->default_data_type());
  // CHECK shape
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 2);
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(in_blob_desc->shape());
  out_blob_desc->set_data_type(conf.out().data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());
  // tmp
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp");
  tmp_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0)});
  tmp_blob_desc->set_data_type(conf.out().data_type());
  tmp_blob_desc->set_has_data_id(false);
}

REGISTER_OP(OperatorConf::kSoftmaxConf, SoftmaxOp);

}  // namespace oneflow
