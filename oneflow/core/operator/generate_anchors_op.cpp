#include "oneflow/core/operator/generate_anchors_op.h"

namespace oneflow {

void GenerateAnchorsOp::InitFromOpConf() {
  CHECK(op_conf().has_generate_anchors_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  EnrollConstBufBn("anchors");

  GenerateAnchorsOpConf* conf = mut_op_conf()->mutable_generate_anchors_conf();
  CHECK_GE(conf->scale_size(), conf->scale_apply_ratio_size());
  CHECK_GE(conf->aspect_ratio_size(), conf->ratio_apply_flip_size());
  FOR_RANGE(size_t, i, conf->scale_apply_ratio_size(), conf->scale_size()) {
    conf->add_scale_apply_ratio(true);
  }
  FOR_RANGE(size_t, i, conf->ratio_apply_flip_size(), conf->aspect_ratio_size()) {
    conf->add_ratio_apply_flip(true);
  }
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

  int64_t num_anchors = 0;
  FOR_RANGE(size_t, i, 0, conf.scale_size()) {
    num_anchors += 1;
    if (conf.scale_apply_ratio(i)) {
      FOR_RANGE(size_t, j, 0, conf.aspect_ratio_size()) {
        num_anchors += 1 + conf.ratio_apply_flip(j);
      }
    }
  }
  // const buf: anchors (H, W, A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({feat_h, feat_w, num_anchors, 4});
  anchors_blob_desc->set_data_type(in_blob_desc->data_type());
  // output: same as anchors
  *GetBlobDesc4BnInOp("anchors") = *anchors_blob_desc;
}

REGISTER_CPU_OP(OperatorConf::kGenerateAnchorsConf, GenerateAnchorsOp);

}  // namespace oneflow