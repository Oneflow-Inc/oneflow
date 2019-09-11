#ifndef ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct SegmentKernelUtilImpl final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                                const K* segment_ids, const int32_t unique_ids_num, T* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const Shape& in_shape, const Shape& id_shape,
                                 const T* out_diff, const K* segment_ids, T* in_diff);
};

template<DeviceType device_type, typename T, typename K>
struct SegmentKernelUtil final {
  static Shape ClipDim0(const Blob* blob) {
    const std::vector<int64_t> dims = blob->shape().dim_vec();
    const std::vector<int64_t> inner_dims(dims.begin() + 1, dims.end());
    return Shape(inner_dims);
  }

  static void SegmentSumForward(DeviceCtx* ctx, const Blob* in, const Blob* segment_ids,
                                const int32_t unique_ids_num, Blob* out) {
    const Shape in_inner_shape = ClipDim0(in);
    const Shape id_inner_shape = ClipDim0(segment_ids);
    const Shape out_inner_shape = ClipDim0(out);

    FOR_RANGE(int32_t, index, 0, in->shape().At(0)) {
      const auto cur_in_ptr = in->dptr<T>() + index * in_inner_shape.elem_cnt();
      const auto cur_id_ptr = segment_ids->dptr<K>() + index * id_inner_shape.elem_cnt();
      auto cur_out_ptr = out->mut_dptr<T>() + index * out_inner_shape.elem_cnt();
      SegmentKernelUtilImpl<device_type, T, K>::SegmentSumForward(
          ctx, in_inner_shape, cur_in_ptr, cur_id_ptr, unique_ids_num, cur_out_ptr);
    }
  }

  static void SegmentSumBackward(DeviceCtx* ctx, const Blob* in, const Blob* segment_ids,
                                 Blob* out) {
    const Shape in_inner_shape = ClipDim0(in);
    const Shape id_inner_shape = ClipDim0(segment_ids);
    const Shape out_inner_shape = ClipDim0(out);

    FOR_RANGE(int32_t, index, 0, in->shape().At(0)) {
      const auto cur_in_ptr = in->dptr<T>() + index * in_inner_shape.elem_cnt();
      const auto cur_id_ptr = segment_ids->dptr<K>() + index * id_inner_shape.elem_cnt();
      auto cur_out_ptr = out->mut_dptr<T>() + index * out_inner_shape.elem_cnt();
      SegmentKernelUtilImpl<device_type, T, K>::SegmentSumBackward(
          ctx, in_inner_shape, id_inner_shape, cur_in_ptr, cur_id_ptr, cur_out_ptr);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SEGMENT_KERNEL_UTIL_H_
