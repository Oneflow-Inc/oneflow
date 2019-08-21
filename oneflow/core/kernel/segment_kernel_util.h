#ifndef ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template <DeviceType device_type, typename T, typename K>
struct SegmentKernelUtilImpl final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                                const K* segment_ids, T* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const Shape& out_diff_shape,
                                const Shape& segment_ids_shape, const T* out_diff, 
                                const K* segment_ids, T* in_diff);
};

template <DeviceType device_type, typename T, typename K>
struct SegmentKernelUtil final {
  static void SegmentSumForward(DeviceCtx* ctx, const Blob* in, const Blob* segment_ids,
                                Blob* out) {
    const Shape& in_shape = in->shape();
    SegmentKernelUtilImpl<device_type, T, K>::SegmentSumForward(ctx, in_shape, in->dptr<T>(),
                                                segment_ids->dptr<K>(), out->mut_dptr<T>());
  }
  static void SegmentSumBackward(DeviceCtx* ctx, const Blob* out_diff, const Blob* segment_ids,
                                 Blob* in_diff) {
    const Shape& segment_ids_shape = segment_ids->shape();
    const Shape& out_diff_shape = out_diff->shape(); 
    SegmentKernelUtilImpl<device_type, T, K>::SegmentSumBackward(ctx, out_diff_shape,
                                              segment_ids_shape, out_diff->dptr<T>(),
                                              segment_ids->dptr<K>(), in_diff->mut_dptr<T>());
  }
};

} // namespace oneflow


#endif // ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
