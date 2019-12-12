#include "oneflow/core/kernel/box_encode_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxEncodeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxEncodeKernel);
  BoxEncodeKernel() = default;
  ~BoxEncodeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* ref_boxes_blob = BnInOp2Blob("ref_boxes");
    const Blob* boxes_blob = BnInOp2Blob("boxes");
    Blob* boxes_delta_blob = BnInOp2Blob("boxes_delta");
    const BBoxRegressionWeights reg_weights =
        this->op_conf().box_encode_conf().regression_weights();

    CHECK_EQ(ref_boxes_blob->shape().NumAxes(), 2);
    CHECK_EQ(ref_boxes_blob->shape().At(1), 4);
    CHECK_EQ(boxes_blob->shape().NumAxes(), 2);
    CHECK_EQ(boxes_blob->shape().At(1), 4);
    const int32_t num_boxes = ref_boxes_blob->shape().At(0);
    CHECK_EQ(num_boxes, boxes_blob->shape().At(0));
    const T* ref_boxes_ptr = ref_boxes_blob->dptr<T>();
    const T* boxes_ptr = boxes_blob->dptr<T>();
    T* boxes_delta_ptr = boxes_delta_blob->mut_dptr<T>();
    BoxEncodeUtil<device_type, T>::Encode(
        ctx.device_ctx, num_boxes, ref_boxes_ptr, boxes_ptr, reg_weights.weight_x(),
        reg_weights.weight_y(), reg_weights.weight_w(), reg_weights.weight_h(), boxes_delta_ptr);
  }
};

template<typename T>
struct BoxEncodeUtil<DeviceType::kCPU, T> {
  static void Encode(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                     const T* boxes_ptr, const float weight_x, const float weight_y,
                     const float weight_w, const float weight_h, T* boxes_delta_ptr) {
    UNIMPLEMENTED();
  }
};

#define REGISTER_BOX_ENCODE_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBoxEncodeConf, dev, dtype, \
                                        BoxEncodeKernel<dev, dtype>)

REGISTER_BOX_ENCODE_KERNEL(DeviceType::kGPU, float);
REGISTER_BOX_ENCODE_KERNEL(DeviceType::kGPU, double);
REGISTER_BOX_ENCODE_KERNEL(DeviceType::kCPU, float);
REGISTER_BOX_ENCODE_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
