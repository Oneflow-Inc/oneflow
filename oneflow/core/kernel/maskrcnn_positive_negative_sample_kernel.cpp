#include "oneflow/core/kernel/maskrcnn_positive_negative_sample_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MaskrcnnPositiveNegativeSampleKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* pos_inds_blob = BnInOp2Blob("pos_inds");
  const Blob* neg_inds_blob = BnInOp2Blob("neg_inds");
  Blob* sampled_pos_inds_blob = BnInOp2Blob("sampled_pos_inds");
  Blob* sampled_neg_inds_blob = BnInOp2Blob("sampled_neg_inds");

  CHECK_EQ(pos_inds_blob->shape().NumAxes(), 1);
  CHECK_EQ(neg_inds_blob->shape().NumAxes(), 1);
  const auto conf = this->op_conf().maskrcnn_positive_negative_sample_conf();
  const int32_t num_pos = std::min(static_cast<T>(conf.total_subsample_num() * conf.pos_fraction()),
                                   static_cast<T>(pos_inds_blob->shape().elem_cnt()));
  const Shape shape = pos_inds_blob->shape();
  const int32_t num_neg = std::min(static_cast<T>(conf.total_subsample_num() - num_pos),
                                   static_cast<T>(neg_inds_blob->shape().elem_cnt()));
  AutoMemcpy(ctx.device_ctx, sampled_pos_inds_blob->mut_dptr<T>(), pos_inds_blob->dptr<T>(),
             num_pos * sizeof(T), sampled_pos_inds_blob->mem_case(), pos_inds_blob->mem_case());
  AutoMemcpy(ctx.device_ctx, sampled_neg_inds_blob->mut_dptr<T>(), neg_inds_blob->dptr<T>(),
             num_neg * sizeof(T), sampled_neg_inds_blob->mem_case(), neg_inds_blob->mem_case());
  sampled_pos_inds_blob->set_dim0_valid_num(0, num_pos);
  sampled_neg_inds_blob->set_dim0_valid_num(0, num_neg);
}

template<DeviceType device_type, typename T>
void MaskrcnnPositiveNegativeSampleKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {
  // already done in ForwardDataContent
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaskrcnnPositiveNegativeSampleConf,
                           MaskrcnnPositiveNegativeSampleKernel, INT_DATA_TYPE_SEQ);

}  // namespace oneflow
