#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {

};

} // namespace oneflow

template <typename T, typename K>
struct SegmentKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                               const K* segment_ids, T* out);
};

template <typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kGPU, T, K>::SegmentSumForward(DeviceCtx* ctx, 
                                                    const Shape& data_shape, const T* in,
                                                    const K* segment_ids, T* out){
  const auto config = 
  // output_rows : unique segment ids
  // segment_ids shape
  // segment_ids
  // data_size : size of input data tensor
  // data : input data tensor
  // output : output reshape to {output_rows, output.size / output_rows}
  const int32_t input_total_size = data_shape.elem_cnt();
  const int32_t input_outer_dim_size = segment_ids_shape.At(0);
  const int32_t input_inner_dim_size = input_total_size / input_outer_dim_size;


}
