#include "oneflow/core/kernel/non_maximum_suppression_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NonMaximumSuppressionKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& op_conf = this->op_conf().non_maximum_suppression_conf();
  const Blob* boxes_blob = BnInOp2Blob("in");
  Blob* suppression_blob = BnInOp2Blob("fw_tmp");
  Blob* keep_blob = BnInOp2Blob("out");
  size_t num_boxes = boxes_blob->shape().At(0);
  size_t num_keep = std::min<size_t>(num_boxes, op_conf.post_nms_top_n());
  NonMaximumSuppressionUtil<device_type, T>::Forward(
      ctx.device_ctx, num_boxes, op_conf.nms_iou_threshold(), num_keep, boxes_blob->dptr<T>(),
      suppression_blob->mut_dptr<int64_t>(), keep_blob->mut_dptr<int8_t>());
}

template<DeviceType device_type, typename T>
void NonMaximumSuppressionKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<typename T>
struct NonMaximumSuppressionUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const size_t num_boxes, const float nms_iou_threshold,
                      const size_t num_keep, const T* boxes, int64_t* suppression, int8_t* keep) {
    TODO();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNonMaximumSuppressionConf, NonMaximumSuppressionKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
