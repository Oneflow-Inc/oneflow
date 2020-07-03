#ifndef ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename I, size_t NDims>
struct ArgWhereKernelUtil {
  static void ArgWhere(DeviceCtx* ctx, const ShapeView& in_shape, const T* in_ptr, void* tmp,
                       size_t tmp_max_bytes, I* out_ptr, I* out_size_ptr);
  static size_t GetArgWhereWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n);
};

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, dtype, itype, ndims) \
  template struct ArgWhereKernelUtil<device_type_v, dtype, itype, ndims>;

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL(device_type_v, dtype_pair, itype_pair)          \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 1)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 2)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 3)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 4)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 5)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
