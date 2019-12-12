#include "oneflow/core/kernel/identify_non_small_boxes_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class IdentifyNonSmallBoxesKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentifyNonSmallBoxesKernel);
  IdentifyNonSmallBoxesKernel() = default;
  ~IdentifyNonSmallBoxesKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    CHECK_EQ(in_blob->shape().NumAxes(), 2);
    CHECK_EQ(in_blob->shape().At(1), 4);
    IdentifyNonSmallBoxesUtil<device_type, T>::IdentifyNonSmallBoxes(
        ctx.device_ctx, in_blob->dptr<T>(), in_blob->shape().At(0),
        this->op_conf().identify_non_small_boxes_conf().min_size(),
        BnInOp2Blob("out")->mut_dptr<int8_t>());
  }
};

template<typename T>
struct IdentifyNonSmallBoxesUtil<DeviceType::kCPU, T> {
  static void IdentifyNonSmallBoxes(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                                    const float min_size, int8_t* out_ptr) {
    FOR_RANGE(int32_t, i, 0, num_boxes) {
      if (in_ptr[i * 4 + 2] - in_ptr[i * 4] >= min_size
          && in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] >= min_size) {
        out_ptr[i] = 1;
      }
    }
  }
};

#define REGISTER_IDENTIFY_NON_SMALL_BOXES_KERNEL(dev, dtype)                                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kIdentifyNonSmallBoxesConf, dev, dtype, \
                                        IdentifyNonSmallBoxesKernel<dev, dtype>)

REGISTER_IDENTIFY_NON_SMALL_BOXES_KERNEL(DeviceType::kGPU, float);
REGISTER_IDENTIFY_NON_SMALL_BOXES_KERNEL(DeviceType::kGPU, double);
REGISTER_IDENTIFY_NON_SMALL_BOXES_KERNEL(DeviceType::kCPU, float);
REGISTER_IDENTIFY_NON_SMALL_BOXES_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
