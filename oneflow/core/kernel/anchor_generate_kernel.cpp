#include "oneflow/core/kernel/anchor_generate_kernel.h"

namespace oneflow {

template<typename T>
void AnchorGenerateKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int64_t batch_height = images_blob->shape().At(1);
  const int64_t batch_width = images_blob->shape().At(2);

  // output: anchors
  Blob* anchors = BnInOp2Blob("anchors");
  Memset<DeviceType::kCPU>(ctx.device_ctx, anchors->mut_dptr<T>(), 0,
                           anchors->ByteSizeOfDataContentField());
  const float fm_stride = static_cast<float>(conf.feature_map_stride());
  const int32_t feature_map_height = std::ceil(static_cast<float>(batch_height) / fm_stride);
  const int32_t feature_map_width = std::ceil(static_cast<float>(batch_width) / fm_stride);
  auto scales_vec = PbRf2StdVec(conf.anchor_scales());
  auto ratios_vec = PbRf2StdVec(conf.aspect_ratios());
  const size_t num_anchors =
      BBoxUtil<MutBBox>::GenerateAnchors(fm_stride, feature_map_height, feature_map_width,
                                         scales_vec, ratios_vec, anchors->mut_dptr<T>());
  CHECK_LE(num_anchors, anchors->static_shape().At(0));
}

template<typename T>
void AnchorGenerateKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO: get dim0_valid_num from op_ctx?
  const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int64_t batch_height = images_blob->shape().At(1);
  const int64_t batch_width = images_blob->shape().At(2);
  CHECK_EQ(batch_height, images_blob->instance_shape().At(0));
  CHECK_EQ(batch_width, images_blob->instance_shape().At(1));

  const int64_t num_anchors_per_cell = conf.anchor_scales_size() * conf.aspect_ratios_size();
  const float fm_stride = conf.feature_map_stride();
  const int64_t fm_height = std::ceil(batch_height / fm_stride);
  const int64_t fm_width = std::ceil(batch_width / fm_stride);
  BnInOp2Blob("anchors")->set_dim0_valid_num(0, fm_height * fm_width * num_anchors_per_cell);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorGenerateConf, AnchorGenerateKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
