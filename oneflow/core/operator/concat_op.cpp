#include "oneflow/core/operator/concat_op.h"

namespace oneflow {

void ConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_concat_conf());

  for (int i = 0; i < op_conf().concat_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    CHECK(ibn2lbn_.emplace(ibn, op_conf().concat_conf().in(i)).second);
    EnrollInputBn(ibn);
  }
  EnrollOutputBn("out");
}

const PbMessage& ConcatOp::GetSpecialConf() const {
  return op_conf().concat_conf();
}

void ConcatOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  CHECK_EQ(in_0_blob_desc->data_type(), conf.data_type());
  bool has_data_id = in_0_blob_desc->has_data_id();
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  int32_t concat_axis = conf.axis();
  if (concat_axis < 0) { concat_axis += out_dim_vec.size(); }
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().at(i));
    for (int64_t j = 0; j < in_i_blob_desc->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        out_dim_vec[j] += in_i_blob_desc->shape().At(j);
      } else {
        CHECK_EQ(out_dim_vec[j], in_i_blob_desc->shape().At(j));
      }
    }
    CHECK_EQ(in_i_blob_desc->data_type(), conf.data_type());
    CHECK_EQ(has_data_id, in_i_blob_desc->has_data_id());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id(has_data_id);
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
