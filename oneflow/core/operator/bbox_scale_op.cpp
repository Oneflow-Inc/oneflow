#include "oneflow/core/operator/bbox_scale_op.h"

namespace oneflow {

void BboxScaleOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_scale_conf());
  EnrollInputBn("in", false);
  EnrollInputBn("scale", false);
  EnrollOutputBn("out", false);
}

void BboxScaleOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: in (N, R, 4) or (R, 4)
  const BlobDesc* in_box = GetBlobDesc4BnInOp("in");
  // input: scale (N, 2)
  const BlobDesc* scale = GetBlobDesc4BnInOp("scale");

  const int64_t dims = in_box->shape().NumAxes();
  CHECK_EQ(in_box->shape().At(dims - 1), 4);
  CHECK_EQ(scale->shape().NumAxes(), 2);
  CHECK_EQ(scale->shape().At(1), 2);
  if (dims == 2) {
    CHECK(in_box->has_record_id_in_device_piece_field());
  } else if (dims == 3) {
    CHECK(!in_box->has_dim0_valid_num_field());
    CHECK(in_box->has_dim1_valid_num_field());
    CHECK_EQ(scale->shape().At(0), in_box->shape().At(0));
  } else {
    UNIMPLEMENTED();
  }
  CHECK_EQ(in_box->data_type(), scale->data_type());

  // output: out (N, R, 4) or (R, 4)
  *GetBlobDesc4BnInOp("out") = *in_box;
}

REGISTER_CPU_OP(OperatorConf::kBboxScaleConf, BboxScaleOp);

}  // namespace oneflow
