#include "oneflow/core/operator/generate_anchors_op.h"

namespace oneflow {

void GenerateAnchorsOp::InitFromOpConf() {
  CHECK(op_conf().has_generate_anchors_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  EnrollConstBufBn("anchors");
}

const PbMessage& GenerateAnchorsOp::GetCustomizedConf() const {
  return op_conf().generate_anchors_conf();
}

void GenerateAnchorsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const GenerateAnchorsOpConf& conf = op_conf().generate_anchors_conf();
  // input: feature_map (N, C, H, W) T
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const int64_t feat_h = in_blob_desc->shape().At(2);
  const int64_t feat_w = in_blob_desc->shape().At(3);
  const int32_t img_h = conf.image_height();
  const int32_t img_w = conf.image_height();
  const float feat_step = conf.feature_map_stride();
  CHECK_EQ(feat_h, static_cast<int64_t>(std::ceil(img_h / feat_step)));
  CHECK_EQ(feat_w, static_cast<int64_t>(std::ceil(img_w / feat_step)));
  // num of anchors
  CHECK((conf.scale_conf_size() == 0) || (conf.aspect_ratio_conf_size() == 0));
  int64_t num_anchors = 0;
  for (const AnchorScaleConf& scale_conf : conf.scale_conf()) {
    CHECK_GT(scale_conf.ratio_size(), 0);
    num_anchors += scale_conf.ratio_size();
  }
  for (const AnchorAspectRatioConf& aspect_ratio_conf : conf.aspect_ratio_conf()) {
    CHECK_GT(aspect_ratio_conf.scale_size(), 0);
    num_anchors += aspect_ratio_conf.scale_size();
  }
  // const buf: anchors (H, W, A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({feat_h, feat_w, num_anchors, 4});
  anchors_blob_desc->set_data_type(in_blob_desc->data_type());
  // output: out (N, H, W, A * 4)
  BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
  out_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), feat_h, feat_w, num_anchors * 4});
  out_desc->set_data_type(in_blob_desc->data_type());
}

REGISTER_CPU_OP(OperatorConf::kGenerateAnchorsConf, GenerateAnchorsOp);

}  // namespace oneflow