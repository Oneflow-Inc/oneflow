#include "oneflow/core/operator/anchor_generate_op.h"

namespace oneflow {

void AnchorGenerateOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_generate_conf());
  EnrollInputBn("images", false);
  EnrollRepeatedOutputBn("anchors", false);
}

const PbMessage& AnchorGenerateOp::GetCustomizedConf() const {
  return this->op_conf().anchor_generate_conf();
}

void AnchorGenerateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  const int32_t num_layers = conf.anchor_generator_conf_size();
  // input: images (N, H, W, C)
  const BlobDesc* images = GetBlobDesc4BnInOp("images");
  const int32_t batch_height = images->shape().At(1);
  const int32_t batch_width = images->shape().At(2);
  CHECK_GT(batch_height, 0);
  CHECK_GT(batch_width, 0);

  FOR_RANGE(size_t, layer, 0, num_layers) {
    const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf(layer);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    CHECK_GT(fm_stride, 0);
    const int64_t fm_height = std::ceil(batch_height / fm_stride);
    const int64_t fm_width = std::ceil(batch_width / fm_stride);
    CHECK_GT(fm_height, 0);
    CHECK_GT(fm_width, 0);
    int64_t num_anchors_per_layer = fm_height * fm_width * num_anchors_per_cell;
    // repeat output: anchors (h_i * w_i * A, 4) int32_t
    BlobDesc* anchors = GetBlobDesc4BnInOp(GenRepeatedBn("anchors", layer));
    anchors->set_data_type(images->data_type());
    anchors->mut_shape() = Shape({num_anchors_per_layer, 4});
    anchors->set_has_dim0_valid_num_field(true);
    anchors->mut_dim0_inner_shape() = Shape({1, num_anchors_per_layer});
  }
}

REGISTER_CPU_OP(OperatorConf::kAnchorGenerateConf, AnchorGenerateOp);

}  // namespace oneflow
