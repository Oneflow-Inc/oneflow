#include "oneflow/core/kernel/norm_kernel.h"
#include "oneflow/core/kernel/norm_kernel.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO(shiyuan)
  const NormOpConf& conf = this->op_conf().norm_conf();
  CHECK_EQ(conf.axis(), 0);
  CHECK_EQ(conf.norm_type(), Norm::kL1);

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* abs_tmp_blob = BnInOp2Blob("abs_tmp");
  Blob* sum_tmp_blob = BnInOp2Blob("sum_tmp");
  Blob* out_blob = BnInOp2Blob("out");
  NormKernelUtil<device_type, T>::Abs(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                      static_cast<T>(conf.epsilon()), in_blob->dptr<T>(),
                                      abs_tmp_blob->mut_dptr<T>());
  int32_t offset = in_blob->shape().elem_cnt() / out_blob->shape().elem_cnt();
  for (int32_t i = 0; i < out_blob->shape().elem_cnt(); ++i) {
    KernelUtil<device_type, T>::Sum(ctx.device_ctx, offset, abs_tmp_blob->dptr<T>() + i * offset,
                                    out_blob->mut_dptr<T>() + i, sum_tmp_blob->mut_dptr<T>(),
                                    sum_tmp_blob->ByteSizeOfDataContentField());
  }
}

template<DeviceType device_type, typename T>
void NormKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  int32_t in_n = in_blob->shape().elem_cnt();
  NormKernelUtil<device_type, T>::L1NormBackward(ctx.device_ctx, in_n, in_blob->shape().Count(1),
                                                 out_diff_blob->dptr<T>(), in_blob->dptr<T>(),
                                                 in_diff_blob->mut_dptr<T>());
}

template<typename T>
struct NormKernelUtil<DeviceType::kCPU, T> {
  static void Abs(DeviceCtx* ctx, const int32_t n, const T epsilon, const T* in_dptr,
                  T* abs_tmp_dptr) {
    for (int32_t i = 0; i < n; ++i) { abs_tmp_dptr[i] = std::abs(in_dptr[i]) + epsilon; }
  }

  static void L1NormBackward(DeviceCtx* ctx, const int32_t in_n, const int32_t offset,
                             const T* out_diff_dptr, const T* in_dptr, T* in_diff_dptr) {
    FOR_RANGE(int32_t, i, 0, in_n) {
      in_diff_dptr[i] = L1NormInDiff(out_diff_dptr[i / offset], in_dptr[i]);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormConf, NormKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
