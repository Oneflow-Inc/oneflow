#include "oneflow/core/kernel/bbox_transform_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BboxTransformKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  const BBoxRegressionWeights& bbox_reg_ws =
      this->op_conf().bbox_transform_conf().bbox_reg_weights();
  const int64_t num_axes = out_bbox_blob->shape().NumAxes();
  const int64_t num_boxes = out_bbox_blob->shape().Count(0, num_axes - 1);
  const int64_t num_classes =
      bbox_delta_blob->shape().At(num_axes - 1) / bbox_blob->shape().At(num_axes - 1);
  FOR_RANGE(int64_t, i, 0, num_boxes) {
    FOR_RANGE(int64_t, j, 0, num_classes) {
      int32_t offset = i * num_classes + j;
      const auto* bbox = BBox::Cast(bbox_blob->dptr<T>()) + i;
      const auto* bbox_delta = BBoxDelta<T>::Cast(bbox_delta_blob->dptr<T>()) + offset;
      auto* out_bbox = BBox::MutCast(out_bbox_blob->mut_dptr<T>()) + offset;
      out_bbox->Transform(bbox, bbox_delta, bbox_reg_ws);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxTransformConf, BboxTransformKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow