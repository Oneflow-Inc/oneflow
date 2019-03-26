#include "oneflow/core/operator/ground_truth_bbox_scale_op.h"

namespace oneflow {

void GtBboxScaleOp::InitFromOpConf() {
  CHECK(op_conf().has_gt_bbox_scale_conf());
  EnrollInputBn("in", false);
  EnrollInputBn("scale", false);
  EnrollOutputBn("out", false);
}

void GtBboxScaleOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_gt_bbox = GetBlobDesc4BnInOp("in");
  const BlobDesc* scale = GetBlobDesc4BnInOp("scale");

  CHECK_EQ(in_gt_bbox->shape().At(0), scale->shape().At(0));
  CHECK(in_gt_bbox->has_dim1_valid_num_field());
  CHECK_EQ(scale->shape().At(1), 2);
  CHECK_EQ(in_gt_bbox->data_type(), scale->data_type());

  *GetBlobDesc4BnInOp("out") = *in_gt_bbox;
}

void GtBboxScaleOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("scale")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kGtBboxScaleConf, GtBboxScaleOp);

}  // namespace oneflow
