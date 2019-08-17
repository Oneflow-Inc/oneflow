#ifndef ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_

namespace oneflow {

template <DeviceType device_type, typename T, typename K>
struct SegmentKernelUtil final {
  static void SegmentSumForward(DeviceCtx* ctx, const Blob* in, const Blob* segment_ids, Blob* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const Blob* out_diff, const Blob* segment_ids,
                                 Blob* in_diff);
};

template <DeviceType device_type, typename T, typename K>
struct SegmentKernelUtilImpl final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                                const K* segment_ids, T* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const Shape& out_diff_shape, const T* out_diff,
                                 const K* segment_ids, T* in_diff);
};

}


#endif // ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
