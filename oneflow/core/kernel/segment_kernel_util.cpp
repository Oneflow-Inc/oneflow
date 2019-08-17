#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {
  

} // namespace oneflow 

template <DeviceType device_type, typename T, typename K>
void SegmentKernelUtil::SegmentSumForward(DeviceCtx* ctx, const Blob* in, const Blob* segment_ids,
                                          Blob* out) {
  const Shape& in_shape = in->shape();
  const Shape& ids_shape = in->shape();
  // segment_ids must be 1-D tensor and rank euqal to the data's 1st dimension
  CHECK(ids_shape.NumAxes() == 1);
  CHECK(in_shape->At(0) == ids_shape.At(0));

  SegmentKernelUtilImpl::SegmentSumForward(ctx, in_shape, in->dptr<T>(), segment_ids->dptr<K>,
                                           out->dptr<T>());
}


template<typename T, typename K>
struct SegmentKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                                const K* segment_ids, T* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const int32_t segment_ids_count, const T* out_diff,
                                 const K* segment_ids, T* in_diff);
};


template <typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kCPU, T, K>::SegmentSumForward(DeviceCtx* ctx,
                                                    const Shape& in_shape, const T* in, 
                                                    const K* segment_ids, T* out) {
  const int32_t batch_size = in_shape.At(0);
  const int32_t axes_num = in_shape.NumAxes();
  const int32_t row_offset = in_shape.Count(1, axes_num);
  int32_t current_id = segment_ids[0];
  // segment ids must start from 0 
  CHECK(current_id == 0);
  std::copy(in, in+row_offset, out);

  FOR_RANGE(int32_t, idx, 1, batch_size) {
    // segment ids are increasing
    CHECK_GE(segment_ids[idx], current_id);
    auto in_offset = idx * row_offset;
    auto out_offset = current_id * row_offset;
    if (current_id == segment_ids[idx]){
      VectorAdd(out + out_offset, in + in_offset, row_offset, out + out_offset);
    }
    else{
      ++current_id;
      auto from = in + in_offset;
      auto to = out + out_offset;
      std::copy(from, from + row_offset, to);
    }
  }
}

template <typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kCPU, T, K>::SegmentSumBackward(DeviceCtx* ctx,
                                                    const int32_t segment_ids_count, const T* out_diff,
                                                    const K* segment_ids, T* in_diff){
  const int32_t axes_num = out_diff_shape.NumAxes();
  const int32_t row_offset = out_diff.Count(1, axes_num);
  int32_t current_id = segment_ids[0];
  // segment ids must start from 0
  CHECK(current_id == 0);
  std::copy(out_diff, out_diff + row_offset, in_diff);

  FOR_RANGE(int32_t, idx, 1, segment_ids_count){
    // segment ids are increasing
    CHECK_GE(segment_ids[idx], current_id);
    auto to = in_diff + idx * row_offset;
    if (current_id == segment_ids[idx]){
      auto from = out_diff + current_id * row_offset;
      std::copy(from, from + row_offset, to); 
    }
    else{
      ++current_id;
      auto from = out_diff + current_id * row_offset;
      std::copy(from, from + row_offset, to);
    }
  }
}


template <typename T>
inline void VectorAdd(const T* lhs, const T* rhs, int32_t elems_count, T* out){
  FOR_RANGE(int32_t, idx, 0, elems_count){
    out[idx] = lhs[idx] + rhs[idx];
  }
}

