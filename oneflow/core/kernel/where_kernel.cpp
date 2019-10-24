#include "oneflow/core/kernel/where_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void WhereKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* condition_blob = BnInOp2Blob("condition");
  int64_t elem_cnt = condition_blob->shape().elem_cnt();

  WhereKernelUtil<device_type, T>::Where(ctx.device_ctx, elem_cnt, condition_blob->dptr<T>(),
                                         BnInOp2Blob("x")->dptr<T>(), BnInOp2Blob("y")->dptr<T>(),
                                         BnInOp2Blob("out")->mut_dptr<T>());
}

template<typename T>
struct WhereKernelUtil<DeviceType::kCPU, T> {
  static void Where(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* x_dptr,
                    const T* y_dptr, T* out_dptr) {
    FOR_RANGE(int64_t, i, 0, n) {
      out_dptr[i] = (cond_dptr[i] != 0) * x_dptr[i] + (cond_dptr[i] == 0) * y_dptr[i];
    }
  }
  static void CmptXDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* x_diff_dptr) {
    FOR_RANGE(int64_t, i, 0, n) { x_diff_dptr[i] = (cond_dptr[i] != 0) * out_diff_dptr[i]; }
  }
  static void CmptYDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* y_diff_dptr) {
    FOR_RANGE(int64_t, i, 0, n) { y_diff_dptr[i] = (cond_dptr[i] == 0) * out_diff_dptr[i]; }
  }
};

#define INSTANTIATE_WHERE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct WhereKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_WHERE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kWhereConf, WhereKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
