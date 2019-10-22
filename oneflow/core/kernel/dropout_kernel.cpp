#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t elem_cnt = BnInOp2Blob("in")->shape().elem_cnt();
  if (this->job_desc().IsTrain()) {
    DropoutKernelUtil<device_type, T>::MaskAndScale(
        ctx.device_ctx, elem_cnt, this->op_conf().dropout_conf().scale(),
        BnInOp2Blob("in")->dptr<T>(), BnInOp2Blob("mask")->dptr<int8_t>(),
        BnInOp2Blob("out")->mut_dptr<T>());
  } else {
    Memcpy<device_type>(ctx.device_ctx, BnInOp2Blob("out")->mut_dptr<void>(),
                        BnInOp2Blob("in")->dptr<void>(), elem_cnt * sizeof(T));
  }
}

template<typename T>
struct DropoutKernelUtil<DeviceType::kCPU, T> final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x,
                           const int8_t* mask, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
  }
};

#define INITIATE_DROPOUT_KERNEL_UTIL_CPU(T, type_proto) \
  template struct DropoutKernelUtil<DeviceType::kCPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_DROPOUT_KERNEL_UTIL_CPU,
                     ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef INITIATE_DROPOUT_KERNEL_UTIL_CPU

#define REGISTER_DROPOUT_KERNEL(dev, dtype)                                     \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDropoutConf, dev, dtype, \
                                        DropoutKernel<dev, dtype>)

REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, float);
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, double);
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, int8_t);
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, int64_t);
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, float16);
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, float);
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, double);
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, int8_t);
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, int32_t);
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, int64_t);
}  // namespace oneflow
