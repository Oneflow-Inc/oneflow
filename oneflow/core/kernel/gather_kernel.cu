#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

namespace {}  // namespace

template<typename T>
struct LookUpKernelUtil<DeviceType::kGPU, T> final {
  static void Forward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices, const T* in,
                      int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const int32_t* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_rows, int64_t in_cols, T* in_diff);
};

template<typename T>
void LookUpKernelUtil<DeviceType::kGPU, T>::Forward(DeviceCtx* ctx, const int32_t* indices,
                                                    int64_t num_indices, const T* in,
                                                    int64_t in_rows, int64_t in_cols, T* out) {}

template<typename T>
void LookUpKernelUtil<DeviceType::kGPU, T>::Backward(DeviceCtx* ctx, const int32_t* indices,
                                                     int64_t num_indices, const T* out_diff,
                                                     int64_t in_rows, int64_t in_cols, T* in_diff) {

}

#define INITIATE_LOOK_UP_KERNEL_UTIL(T, type_proto) \
  template struct LookUpKernelUtil<DeviceType::kGPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_LOOK_UP_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);
#undef INITIATE_LOOK_UP_KERNEL_UTIL

}  // namespace oneflow
