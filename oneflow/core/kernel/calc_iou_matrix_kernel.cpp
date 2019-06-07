#include "oneflow/core/kernel/calc_iou_matrix_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void CalcIoUMatrixKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* boxes1_blob = BnInOp2Blob("boxes1");
  const Blob* boxes2_blob = BnInOp2Blob("boxes2");
  Blob* iou_matrix_blob = BnInOp2Blob("iou_matrix");

  CHECK_EQ(boxes1_blob->shape().NumAxes(), 2);
  CHECK_EQ(boxes1_blob->shape().At(1), 4);
  CHECK_EQ(boxes2_blob->shape().NumAxes(), 2);
  CHECK_EQ(boxes2_blob->shape().At(1), 4);
  const int32_t num_boxes1 = boxes1_blob->shape().At(0);
  const int32_t num_boxes2 = boxes2_blob->shape().At(0);
  const T* boxes1_ptr = boxes1_blob->dptr<T>();
  const T* boxes2_ptr = boxes2_blob->dptr<T>();
  float* iou_matrix_ptr = iou_matrix_blob->mut_dptr<float>();
  CalcIoUMatrixUtil<device_type, T>::Forward(ctx.device_ctx, boxes1_ptr, num_boxes1, boxes2_ptr,
                                             num_boxes2, iou_matrix_ptr);
}

template<DeviceType device_type, typename T>
void CalcIoUMatrixKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* boxes1_blob = BnInOp2Blob("boxes1");
  CHECK(boxes1_blob->has_dim0_valid_num_field());
  BnInOp2Blob("iou_matrix")->set_dim0_valid_num(0, boxes1_blob->shape().At(0));
}

template<DeviceType device_type, typename T>
void CalcIoUMatrixKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* boxes2_blob = BnInOp2Blob("boxes2");
  CHECK(boxes2_blob->has_dim0_valid_num_field());
  BnInOp2Blob("iou_matrix")->set_instance_shape(Shape({boxes2_blob->shape().At(0)}));
}

template<typename T>
struct CalcIoUMatrixUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const T* boxes1_ptr, const int32_t num_boxes1,
                      const T* boxes2_ptr, const int32_t num_boxes2, float* iou_matrix_ptr) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalcIouMatrixConf, CalcIoUMatrixKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
