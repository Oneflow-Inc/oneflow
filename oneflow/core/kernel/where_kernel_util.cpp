#include "oneflow/core/kernel/where_kernel_util.h"

namespace oneflow {

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

}  // namespace oneflow
