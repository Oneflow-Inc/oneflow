#include "oneflow/core/kernel/upsample_nearest_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void UpsampleNearestKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  UpsampleNearestUtil<device_type, T>::Forward(ctx, 1.f / scale, 1.f / scale, false, in_blob,
                                               out_blob);
}

template<DeviceType device_type, typename T>
void UpsampleNearestGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* dx_blob = BnInOp2Blob("dx");
  if (dx_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                      dx_blob->ByteSizeOfDataContentField());
  const Blob* dy_blob = BnInOp2Blob("dy");
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  UpsampleNearestUtil<device_type, T>::Backward(ctx, 1.f / scale, 1.f / scale, false, dy_blob,
                                                dx_blob);
}

template<typename T>
struct UpsampleNearestUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                       const bool align_corners, const Blob* dy_blob, Blob* dx_blob) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_CPU_UPSAMPLE_NEAREST_UTIL(type_cpp, type_proto) \
  template class UpsampleNearestUtil<DeviceType::kCPU, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_UPSAMPLE_NEAREST_UTIL, FLOATING_DATA_TYPE_SEQ);

#define REGISTER_UPSAMPLE_NEAREST_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kUpsampleNearestConf, dev, dtype, \
                                        UpsampleNearestKernel<dev, dtype>)

REGISTER_UPSAMPLE_NEAREST_KERNEL(DeviceType::kGPU, float);
REGISTER_UPSAMPLE_NEAREST_KERNEL(DeviceType::kGPU, double);
REGISTER_UPSAMPLE_NEAREST_KERNEL(DeviceType::kCPU, float);
REGISTER_UPSAMPLE_NEAREST_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
