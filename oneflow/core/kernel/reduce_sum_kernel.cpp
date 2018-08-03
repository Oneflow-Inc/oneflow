#include "oneflow/core/kernel/reduce_sum_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  if (this->kernel_conf().reduce_sum_conf().has_axis() == false) {
    KernelUtil<device_type, T>::Sum(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                    out_blob->mut_dptr<T>(), fw_tmp_blob->mut_dptr<T>(),
                                    fw_tmp_blob->ByteSizeOfDataContentField());
    return;
  }
  int32_t axis = this->kernel_conf().reduce_sum_conf().axis();
  int64_t lhs_num = in_blob->shape().Count(0, axis);
  int64_t middle_num = in_blob->shape().At(axis);
  int64_t rhs_num = in_blob->shape().Count(axis + 1);
  FOR_RANGE(int64_t, lhs_i, 0, lhs_num) {
    const T* src_ptr = in_blob->dptr<T>() + lhs_i * middle_num * rhs_num;
    T* dst_ptr = out_blob->mut_dptr<T>() + lhs_i * rhs_num;
    Memcpy<device_type>(ctx.device_ctx, dst_ptr, src_ptr, rhs_num * sizeof(T));
    FOR_RANGE(int64_t, middle_i, 1, middle_num) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, rhs_num, 1.0f, src_ptr + middle_i * rhs_num,
                                       1, dst_ptr, 1);
    }
  }
}

template<DeviceType device_type, typename T>
void ReduceSumKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");

  if (this->kernel_conf().reduce_sum_conf().has_axis() == false) {
    T* dst_ptr = in_diff_blob->mut_dptr<T>();
    const T* src_ptr = out_diff_blob->dptr<T>();
    FOR_RANGE(int64_t, i, 0, in_diff_blob->shape().Count(0)) {
      Memcpy<device_type>(ctx.device_ctx, dst_ptr++, src_ptr, sizeof(T));
    }
    return;
  }

  int32_t axis = this->kernel_conf().reduce_sum_conf().axis();
  int64_t lhs_num = in_diff_blob->shape().Count(0, axis);
  int64_t middle_num = in_diff_blob->shape().At(axis);
  int64_t rhs_num = in_diff_blob->shape().Count(axis + 1);
  FOR_RANGE(int64_t, lhs_i, 0, lhs_num) {
    const T* src_ptr = out_diff_blob->dptr<T>() + lhs_i * rhs_num;
    T* dst_ptr = in_diff_blob->mut_dptr<T>() + lhs_i * middle_num * rhs_num;
    FOR_RANGE(int64_t, middle_i, 0, middle_num) {
      Memcpy<device_type>(ctx.device_ctx, dst_ptr, src_ptr, rhs_num * sizeof(T));
      dst_ptr += rhs_num;
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceSumConf, ReduceSumKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
