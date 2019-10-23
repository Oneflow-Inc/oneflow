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

template<DeviceType device_type, typename T>
void DropoutGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DropoutKernelUtil<device_type, T>::MaskAndScale(
      ctx.device_ctx, BnInOp2Blob("dy")->shape().elem_cnt(),
      this->op_conf().dropout_grad_conf().scale(), BnInOp2Blob("dy")->dptr<T>(),
      BnInOp2Blob("mask")->dptr<int8_t>(), BnInOp2Blob("dx")->mut_dptr<T>());
}

template<DeviceType device_type>
void RandomMaskLikeKernel<device_type>::VirtualKernelInit(DeviceCtx* device_ctx) {
  const auto& random_mask_like_conf = this->op_conf().random_mask_like_conf();
  int64_t seed = GetCurTime();
  if (random_mask_like_conf.has_seed()) { seed = random_mask_like_conf.seed(); }
  random_generator_.reset(new RandomGenerator<device_type>(seed, device_ctx));
}

template<DeviceType device_type>
void RandomMaskLikeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->job_desc().IsTrain()) {
    int64_t elem_cnt = BnInOp2Blob("out")->shape().elem_cnt();
    float* random_tmp = BnInOp2Blob("random_tmp")->mut_dptr<float>();
    int8_t* mask = BnInOp2Blob("out")->mut_dptr<int8_t>();
    random_generator_->Uniform(elem_cnt, random_tmp);
    RandomMaskLikeKernelUtil<device_type>::GenMask(
        ctx.device_ctx, elem_cnt, this->op_conf().random_mask_like_conf().rate(), random_tmp, mask);
  } else {
    // do nothing
  }
}

template<typename T>
struct DropoutKernelUtil<DeviceType::kCPU, T> final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x,
                           const int8_t* mask, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
  }
};

template<>
struct RandomMaskLikeKernelUtil<DeviceType::kCPU> final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
                      int8_t* mask) {
    for (int64_t i = 0; i < n; ++i) { mask[i] = random_tmp[i] > threshold; }
  }
};

template struct RandomMaskLikeKernelUtil<DeviceType::kCPU>;

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

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kRandomMaskLikeConf, DeviceType::kCPU,
                            RandomMaskLikeKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kRandomMaskLikeConf, DeviceType::kGPU,
                            RandomMaskLikeKernel<DeviceType::kGPU>);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDropoutGradConf, DropoutGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
