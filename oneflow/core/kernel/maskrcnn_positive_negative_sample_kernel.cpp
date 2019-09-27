#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaskrcnnPositiveNegativeSampleKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnPositiveNegativeSampleKernel);
  MaskrcnnPositiveNegativeSampleKernel() = default;
  ~MaskrcnnPositiveNegativeSampleKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    // already done in ForwardDataContent
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* pos_inds_blob = BnInOp2Blob("pos_inds");
    const Blob* neg_inds_blob = BnInOp2Blob("neg_inds");
    Blob* sampled_pos_inds_blob = BnInOp2Blob("sampled_pos_inds");
    Blob* sampled_neg_inds_blob = BnInOp2Blob("sampled_neg_inds");

    CHECK_EQ(pos_inds_blob->shape().NumAxes(), 1);
    CHECK_EQ(neg_inds_blob->shape().NumAxes(), 1);
    const auto conf = this->op_conf().maskrcnn_positive_negative_sample_conf();
    const int32_t num_pos =
        std::min(static_cast<T>(conf.total_subsample_num() * conf.pos_fraction()),
                 static_cast<T>(pos_inds_blob->shape().elem_cnt()));
    const int32_t num_neg = std::min(static_cast<T>(conf.total_subsample_num() - num_pos),
                                     static_cast<T>(neg_inds_blob->shape().elem_cnt()));
    AutoMemcpy(ctx.device_ctx, sampled_pos_inds_blob->mut_dptr<T>(), pos_inds_blob->dptr<T>(),
               num_pos * sizeof(T), sampled_pos_inds_blob->mem_case(), pos_inds_blob->mem_case());
    AutoMemcpy(ctx.device_ctx, sampled_neg_inds_blob->mut_dptr<T>(), neg_inds_blob->dptr<T>(),
               num_neg * sizeof(T), sampled_neg_inds_blob->mem_case(), neg_inds_blob->mem_case());
    sampled_pos_inds_blob->dense_shape_mut_view().set_shape(Shape({num_pos}));
    sampled_neg_inds_blob->dense_shape_mut_view().set_shape(Shape({num_neg}));
  }
};

#define REGISTER_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL(dtype)                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                          \
      OperatorConf::kMaskrcnnPositiveNegativeSampleConf, DeviceType::kCPU, dtype, \
      MaskrcnnPositiveNegativeSampleKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                          \
      OperatorConf::kMaskrcnnPositiveNegativeSampleConf, DeviceType::kGPU, dtype, \
      MaskrcnnPositiveNegativeSampleKernel<DeviceType::kGPU, dtype>)

REGISTER_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL(int8_t);
REGISTER_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL(int32_t);
REGISTER_MASKRCNN_POSITIVE_NEGATIVE_SAMPLE_KERNEL(int64_t);

}  // namespace oneflow
