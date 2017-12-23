#include "oneflow/core/kernel/sum_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int32_t axis = this->kernel_conf().sum_conf().axis();
  int64_t lhs_num = in_blob->shape().Count(0, axis);
  int64_t middle_num = in_blob->shape().At(axis);
  int64_t rhs_num = in_blob->shape().Count(axis + 1);
  FOR_RANGE(int64_t, lhs_i, 0, lhs_num) {
    const T* src_ptr = in_blob->dptr<T>() + lhs_i * middle_num * rhs_num;
    T* dst_ptr = out_blob->mut_dptr<T>() + lhs_i * rhs_num;
    Memcpy<device_type>(ctx.device_ctx, dst_ptr, src_ptr, rhs_num * sizeof(T));
    FOR_RANGE(int64_t, middle_i, 1, middle_num) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, rhs_num, 1.0f,
                                       src_ptr + middle_i * rhs_num, 1, dst_ptr,
                                       1);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSumConf, SumKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
