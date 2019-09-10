#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {
  
template<typename T, typename K>
struct SegmentKernelUtilImpl<DeviceType::kCPU, T, K> final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape,
                                const T* in, const K* segment_ids, const int32_t unique_ids_num, T* out);

  static void SegmentSumBackward(DeviceCtx* ctx, const Shape& out_diff_shape, 
                                 const Shape& segment_ids_shape, const T* out_diff,
                                 const K* segment_ids, T* in_diff);
};


template <typename T>
inline void VectorAdd(const T* lhs, const T* rhs, int32_t elems_count, T* out){
  FOR_RANGE(int32_t, idx, 0, elems_count){
    out[idx] = lhs[idx] + rhs[idx];
  }
}

template <typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kCPU, T, K>::SegmentSumForward(DeviceCtx* ctx,
                                                    const Shape& in_shape,
                                                    const T* in, 
                                                    const K* segment_ids,
                                                    const int32_t unique_ids_num,
                                                    T* out) {
  // input tensor's dim 0 size
  const int32_t input_outer_dim_size = in_shape.At(0);
  // logically, flatten a n-D tensor to 2-D tensor, 
  // "inner_dim_size" is the cols number of the 2-D tensor
  const int32_t inner_dim_size = in_shape.Count(1, in_shape.NumAxes());
  K last_segment_id = K(-1);

  FOR_RANGE(int32_t, idx, 1, input_outer_dim_size) {
    K current_segment_id = segment_ids[idx];
    if (current_segment_id > last_segment_id) {
      auto from = in + idx * inner_dim_size;
      auto to = out + current_segment_id * inner_dim_size;
      std::copy(from, from + inner_dim_size, to);
    }
    else if (current_segment_id == last_segment_id) {
      auto lhs = out + last_segment_id * inner_dim_size;
      auto rhs = in + idx * inner_dim_size;
      VectorAdd(lhs, rhs, inner_dim_size, lhs);
    }
    last_segment_id = current_segment_id;
  }
}

template <typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kCPU, T, K>::SegmentSumBackward(DeviceCtx* ctx,
                                                    const Shape& out_diff_shape,
                                                    const Shape& segment_ids_shape,
                                                    const T* out_diff,
                                                    const K* segment_ids,
                                                    T* in_diff){
  // in_diff's dim 0 size equals to segment_ids' dim 0 size
  const int32_t in_diff_outer_dim_size = segment_ids_shape.At(0);
  const int32_t inner_dim_size = out_diff_shape.Count(1, out_diff_shape.NumAxes());

  FOR_RANGE(int32_t, idx, 1, in_diff_outer_dim_size){
    const auto current_segment_id = segment_ids[idx];
    const auto to = in_diff + idx * inner_dim_size;
    const auto from = out_diff + current_segment_id * inner_dim_size;
    std::copy(from, from + inner_dim_size, to); 
  }
}

// instantiate template declaration
template struct SegmentKernelUtilImpl<DeviceType::kCPU, float, int32_t>;

} // namespace oneflow
