#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx,
                                                      DeviceCtx* device_ctx) {
  const auto& dropout_conf = this->op_conf().dropout_conf();
  int64_t seed = GetCurTime();
  if (dropout_conf.has_seed()) { seed = dropout_conf.seed(); }
  random_generator_.reset(new RandomGenerator<device_type>(seed, device_ctx));
}

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t elem_cnt = BnInOp2Blob("in")->shape().elem_cnt();
  if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    Dropout(ctx.device_ctx, elem_cnt, this->op_conf().dropout_conf().rate(),
            BnInOp2Blob("in")->dptr<T>(), BnInOp2Blob("random_mask")->mut_dptr<float>(),
            BnInOp2Blob("out")->mut_dptr<T>());
  } else {
    Memcpy<device_type>(ctx.device_ctx, BnInOp2Blob("out")->mut_dptr<void>(),
                        BnInOp2Blob("in")->dptr<void>(), elem_cnt * sizeof(T));
  }
}

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::Dropout(DeviceCtx* ctx, const int64_t n, float dropout_rate,
                                            const T* in, float* random_mask, T* out) const {
  random_generator_->Uniform(n, random_mask);
  DropoutKernelUtil<device_type, T>::MaskAndScale(ctx, n, dropout_rate, 1 / (1 - dropout_rate), in,
                                                  random_mask, out);
}

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::DropoutBackward(DeviceCtx* ctx, const int64_t n,
                                                    float dropout_rate, const T* dy,
                                                    const float* random_mask, T* dx) const {
  DropoutKernelUtil<device_type, T>::MaskAndScale(ctx, n, dropout_rate, 1 / (1 - dropout_rate), dy,
                                                  random_mask, dx);
}

template<typename T>
struct DropoutKernelUtil<DeviceType::kCPU, T> final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float threshold, float scale,
                           const T* x, const float* random_mask, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * (random_mask[i] > threshold) * scale; }
  }
};

#define INITIATE_DROPOUT_KERNEL_UTIL_CPU(T, type_proto) \
  template struct DropoutKernelUtil<DeviceType::kCPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_DROPOUT_KERNEL_UTIL_CPU, ARITHMETIC_DATA_TYPE_SEQ);
#undef INITIATE_DROPOUT_KERNEL_UTIL_CPU

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDropoutConf, DropoutKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
