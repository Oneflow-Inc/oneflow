#include "oneflow/core/kernel/box_decode_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxDecodeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxDecodeKernel);
  BoxDecodeKernel() = default;
  ~BoxDecodeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* ref_boxes_blob = BnInOp2Blob("ref_boxes");
    const Blob* boxes_delta_blob = BnInOp2Blob("boxes_delta");
    Blob* boxes_blob = BnInOp2Blob("boxes");
    const BBoxRegressionWeights reg_weights =
        this->op_conf().box_decode_conf().regression_weights();

    CHECK_EQ(ref_boxes_blob->shape().NumAxes(), 2);
    CHECK_EQ(ref_boxes_blob->shape().At(1), 4);
    CHECK_EQ(boxes_delta_blob->shape().NumAxes(), 2);
    CHECK_EQ(boxes_delta_blob->shape().At(0), ref_boxes_blob->shape().At(0));
    CHECK_EQ(boxes_delta_blob->shape().At(1) % 4, 0);
    const int32_t num_ref_boxes = ref_boxes_blob->shape().At(0);
    const int32_t num_boxes_delta = boxes_delta_blob->shape().elem_cnt() / 4;
    const T* ref_boxes_ptr = ref_boxes_blob->dptr<T>();
    const T* boxes_delta_ptr = boxes_delta_blob->dptr<T>();
    T* boxes_ptr = boxes_blob->mut_dptr<T>();
    BoxDecodeUtil<device_type, T>::Decode(ctx.device_ctx, num_boxes_delta, num_ref_boxes,
                                          ref_boxes_ptr, boxes_delta_ptr, reg_weights.weight_x(),
                                          reg_weights.weight_y(), reg_weights.weight_w(),
                                          reg_weights.weight_h(), boxes_ptr);
  }
};

template<typename T>
struct BoxDecodeUtil<DeviceType::kCPU, T> {
  static void Decode(DeviceCtx* ctx, const int32_t num_boxes_delta, const int32_t num_ref_boxes,
                     const T* ref_boxes_ptr, const T* boxes_delta_ptr, const float weight_x,
                     const float weight_y, const float weight_w, const float weight_h,
                     T* boxes_ptr) {
    UNIMPLEMENTED();
  }
};

#define REGISTER_BOX_DECODE_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBoxDecodeConf, dev, dtype, \
                                        BoxDecodeKernel<dev, dtype>)

REGISTER_BOX_DECODE_KERNEL(DeviceType::kGPU, float);
REGISTER_BOX_DECODE_KERNEL(DeviceType::kGPU, double);
REGISTER_BOX_DECODE_KERNEL(DeviceType::kCPU, float);
REGISTER_BOX_DECODE_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
