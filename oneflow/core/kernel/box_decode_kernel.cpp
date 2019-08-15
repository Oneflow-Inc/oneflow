#include "oneflow/core/kernel/box_decode_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BoxDecodeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* ref_boxes_blob = BnInOp2Blob("ref_boxes");
  const Blob* boxes_delta_blob = BnInOp2Blob("boxes_delta");
  Blob* boxes_blob = BnInOp2Blob("boxes");
  const BBoxRegressionWeights reg_weights = this->op_conf().box_decode_conf().regression_weights();

  CHECK_EQ(ref_boxes_blob->shape().NumAxes(), 2);
  CHECK_EQ(ref_boxes_blob->shape().At(1), 4);
  CHECK_EQ(boxes_delta_blob->shape().NumAxes(), 2);
  CHECK_EQ(boxes_delta_blob->shape().At(1), 4);
  const int32_t num_boxes = ref_boxes_blob->shape().At(0);
  CHECK_EQ(num_boxes, boxes_delta_blob->shape().At(0));
  const T* ref_boxes_ptr = ref_boxes_blob->dptr<T>();
  const T* boxes_delta_ptr = boxes_delta_blob->dptr<T>();
  T* boxes_ptr = boxes_blob->mut_dptr<T>();
  BoxDecodeUtil<device_type, T>::Forward(ctx.device_ctx, num_boxes, ref_boxes_ptr, boxes_delta_ptr,
                                         reg_weights.weight_x(), reg_weights.weight_y(),
                                         reg_weights.weight_w(), reg_weights.weight_h(), boxes_ptr);
}

template<DeviceType device_type, typename T>
void BoxDecodeKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* ref_boxes_blob = BnInOp2Blob("ref_boxes");
  const Blob* boxes_delta_blob = BnInOp2Blob("boxes_delta");
  CHECK(ref_boxes_blob->has_dim0_valid_num_field());
  CHECK(boxes_delta_blob->has_dim0_valid_num_field());
  const int32_t num_boxes = ref_boxes_blob->shape().At(0);
  CHECK_EQ(num_boxes, boxes_delta_blob->shape().At(0));
  BnInOp2Blob("boxes")->set_dim0_valid_num(0, num_boxes);
}

template<typename T>
struct BoxDecodeUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                      const T* boxes_delta_ptr, const float weight_x, const float weight_y,
                      const float weight_w, const float weight_h, T* boxes_ptr) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxDecodeConf, BoxDecodeKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
