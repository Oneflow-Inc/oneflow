#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AnchorGenerateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorGenerateOp);
  AnchorGenerateOp() = default;
  ~AnchorGenerateOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().anchor_generate_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("anchors")->set_value(0);
    return Maybe<void>::Ok();
  }
};

void AnchorGenerateOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_generate_conf());
  EnrollInputBn("images", false);
  EnrollOutputBn("anchors", false);
}

Maybe<void> AnchorGenerateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  // input: images (N, H, W, C)
  const BlobDesc* images = GetBlobDesc4BnInOp("images");
  const int32_t batch_height = images->shape().At(1);
  const int32_t batch_width = images->shape().At(2);
  CHECK_GT_OR_RETURN(batch_height, 0);
  CHECK_GT_OR_RETURN(batch_width, 0);

  // output: anchors (num_anchors, 4)
  const int64_t num_anchors_per_cell = conf.anchor_scales_size() * conf.aspect_ratios_size();
  const float fm_stride = conf.feature_map_stride();
  CHECK_GT_OR_RETURN(fm_stride, 0);
  const int64_t fm_height = std::ceil(batch_height / fm_stride);
  const int64_t fm_width = std::ceil(batch_width / fm_stride);
  CHECK_GT_OR_RETURN(fm_height, 0);
  CHECK_GT_OR_RETURN(fm_width, 0);
  int64_t num_anchors = fm_height * fm_width * num_anchors_per_cell;
  BlobDesc* anchors = GetBlobDesc4BnInOp("anchors");
  anchors->set_data_type(images->data_type());
  anchors->mut_shape() = Shape({num_anchors, 4});
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kAnchorGenerateConf, AnchorGenerateOp);

}  // namespace oneflow
