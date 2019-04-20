#include "oneflow/core/operator/box_scale_op.h"

namespace oneflow {

void BoxScaleOp::InitFromOpConf() {
  CHECK(op_conf().has_box_scale_conf());
  EnrollInputBn("in", false);
  EnrollInputBn("scale", false);
  EnrollOutputBn("out", false);
}

void BoxScaleOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  // input: in_bbox, (N, G, 4) or (R, 4)
  const BlobDesc* in_box = GetBlobDesc4BnInOp("in");
  const bool dim0_varing = in_box->has_dim0_valid_num_field();
  const bool dim1_varing = in_box->has_dim1_valid_num_field();
  CHECK((dim0_varing && !dim1_varing) || (!dim0_varing && dim1_varing));
  if (dim0_varing && !dim1_varing) { CHECK(in_box->has_record_id_in_device_piece_field()); }

  // input: scale, (N, 2)
  const BlobDesc* scale = GetBlobDesc4BnInOp("scale");
  if (!dim0_varing && dim1_varing) { CHECK_EQ(scale->shape().At(0), in_box->shape().At(0)); }
  CHECK_EQ(scale->shape().At(1), 2);
  CHECK_EQ(in_box->data_type(), scale->data_type());

  // output: out_bbox, (N, G, 4) or (R, 4)
  *GetBlobDesc4BnInOp("out") = *in_box;
}

void BoxScaleOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("scale")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kBoxScaleConf, BoxScaleOp);

}  // namespace oneflow
