#include "oneflow/core/kernel/anchor_generate_kernel.h"

namespace oneflow {

template<typename T>
void AnchorGenerateKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int64_t batch_height = images_blob->shape().At(1);
  const int64_t batch_width = images_blob->shape().At(2);

  FOR_RANGE(size_t, i, 0, conf.anchor_generator_conf_size()) {
    Blob* anchors_i_blob = BnInOp2Blob("anchors_" + std::to_string(i));
    Memset<DeviceType::kCPU>(ctx.device_ctx, anchors_i_blob->mut_dptr<T>(), 0,
                             anchors_i_blob->ByteSizeOfDataContentField());

    const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf(i);
    float fm_stride = anchor_generator_conf.feature_map_stride();
    auto scales_vec = PbRf2StdVec(anchor_generator_conf.anchor_scales());
    auto ratios_vec = PbRf2StdVec(anchor_generator_conf.aspect_ratios());
    const size_t num_anchors_per_layer =
        BBoxUtil<MutBBox>::GenerateAnchors(batch_height, batch_width, fm_stride, scales_vec,
                                           ratios_vec, anchors_i_blob->mut_dptr<T>());
    CHECK_LE(num_anchors_per_layer, anchors_i_blob->static_shape().At(0));
  }
}

template<typename T>
void AnchorGenerateKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int64_t batch_height = images_blob->shape().At(1);
  const int64_t batch_width = images_blob->shape().At(2);

  FOR_RANGE(size_t, i, 0, conf.anchor_generator_conf_size()) {
    const auto& anchor_generator_conf = conf.anchor_generator_conf(i);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    const int64_t height = std::ceil(batch_height / fm_stride);
    const int64_t width = std::ceil(batch_width / fm_stride);
    BnInOp2Blob("anchors_" + std::to_string(i))
        ->set_dim0_valid_num(0, height * width * num_anchors_per_cell);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorGenerateConf, AnchorGenerateKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
