#include "oneflow/core/kernel/identify_non_small_boxes_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void IdentifyNonSmallBoxesKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK_EQ(in_blob->shape().NumAxes(), 2);
  CHECK_EQ(in_blob->shape().At(1), 4);
  CHECK(!in_blob->has_instance_shape_field());
  IdentifyNonSmallBoxesUtil<device_type, T>::Forward(
      ctx.device_ctx, in_blob->dptr<T>(), in_blob->shape().At(0),
      this->op_conf().identify_non_small_boxes_conf().min_size(),
      BnInOp2Blob("out")->mut_dptr<int8_t>());
}

template<typename T>
struct IdentifyNonSmallBoxesUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                      const float min_size, int8_t* out_ptr) {
    FOR_RANGE(int32_t, i, 0, num_boxes) {
      if (in_ptr[i * 4 + 2] - in_ptr[i * 4] >= min_size
          && in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] >= min_size) {
        out_ptr[i] = 1;
      }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kIdentifyNonSmallBoxesConf, IdentifyNonSmallBoxesKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
