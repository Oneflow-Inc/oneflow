#include "oneflow/core/kernel/identify_outside_anchors_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void IdentifyOutsideAnchorsKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  IdentifyOutsideAnchorsUtil<device_type, T>::Forward(
      ctx.device_ctx, BnInOp2Blob("anchors"), BnInOp2Blob("image_size"), BnInOp2Blob("out"),
      this->op_conf().identify_outside_anchors_conf().tolerance());
}

template<DeviceType device_type, typename T>
void IdentifyOutsideAnchorsKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("anchors"));
}

template<typename T>
struct IdentifyOutsideAnchorsUtil<DeviceType::kCPU, T> final {
  static void Forward(DeviceCtx* ctx, const Blob* anchors_blob, const Blob* image_size_blob,
                      Blob* identification_blob, float tolerance) {
    int32_t num_anchors = anchors_blob->shape().At(0);
    const T* anchors_dptr = anchors_blob->dptr<T>();
    const int32_t* image_size_dptr = image_size_blob->dptr<int32_t>();
    int8_t* identification_dptr = identification_blob->mut_dptr<int8_t>();
    Memset<DeviceType::kCPU>(ctx, identification_dptr, 0,
                             identification_blob->ByteSizeOfDataContentField());
    FOR_RANGE(int32_t, i, 0, num_anchors) {
      const T* cur_anchor_ptr = anchors_dptr + i * 4;
      if (cur_anchor_ptr[0] < -tolerance || cur_anchor_ptr[1] < -tolerance
          || cur_anchor_ptr[2] >= image_size_dptr[1] + tolerance
          || cur_anchor_ptr[3] >= image_size_dptr[0] + tolerance) {
        identification_dptr[i] = 1;
      }
    }
  }
};

}  // namespace oneflow
