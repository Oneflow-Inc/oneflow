#include <set>
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/segment_kernel_util.h"
#include "oneflow/core/kernel/segment_util.h"

namespace oneflow {

template<typename T, typename K, int OuterDimTileSize>
__global__ void SegmentSumKernel(const int input_outer_dim_size, const int inner_dim_size,
                                 const int output_outer_dim_size, const int total_stripe_count,
                                 const T* input, const K* segment_ids, T* output) {
  for (int stripe_idx = threadIdx.x + blockIdx.x * blockDim.x; stripe_idx < total_stripe_count;
       stripe_idx += gridDim.x * blockDim.x) {
    const int segment_offset = stripe_idx % inner_dim_size;
    const int input_outer_dim_index_base = stripe_idx / inner_dim_size * OuterDimTileSize;
    int first_segment_id = segment_ids[input_outer_dim_index_base];
    int last_segment_id = output_outer_dim_size;

    const int actual_stripe_height =
        min(OuterDimTileSize, input_outer_dim_size - input_outer_dim_index_base);
    T sum = T(0);
    for (int i = 0; i < actual_stripe_height; ++i) {
      K current_segment_id = segment_ids[input_outer_dim_index_base + i];
      if (current_segment_id > last_segment_id) {
        const int output_index = last_segment_id * inner_dim_size + segment_offset;
        if (last_segment_id == first_segment_id) {
          atomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += input[(input_outer_dim_index_base + i) * inner_dim_size + segment_offset];
      last_segment_id = current_segment_id;
    }
    const int output_index = last_segment_id * inner_dim_size + segment_offset;
    atomicAdd(output + output_index, sum);
  }
}

template<typename T, typename K, int OuterDimTileSize>
__global__ void SegmentSumGradKernel(const int in_diff_outer_dim_size, const int inner_dim_size,
                                     const int total_stripe_count, const T* out_diff,
                                     const K* segment_ids, T* in_diff) {
  for (int stripe_idx = threadIdx.x + blockIdx.x * blockDim.x; stripe_idx < total_stripe_count;
       stripe_idx += gridDim.x * blockDim.x) {
    const int segment_offset = stripe_idx % inner_dim_size;
    const int in_diff_outer_dim_index_base = stripe_idx / inner_dim_size * OuterDimTileSize;

    const int actual_stripe_height =
        min(OuterDimTileSize, in_diff_outer_dim_size - in_diff_outer_dim_index_base);

    for (int i = 0; i < actual_stripe_height; ++i) {
      const int logical_row = in_diff_outer_dim_index_base + i;
      const int indiff_index = logical_row * inner_dim_size + segment_offset;
      const int outdiff_index = segment_ids[logical_row] * inner_dim_size + segment_offset;
      in_diff[indiff_index] = out_diff[outdiff_index];
    }
  }
}

template<typename T, typename K>
struct SegmentKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void SegmentSumForward(DeviceCtx* ctx, const Shape& in_shape, const T* in,
                                const K* segment_ids, const int32_t unique_ids_num, T* out);
  static void SegmentSumBackward(DeviceCtx* ctx, const Shape& out_diff_shape,
                                 const Shape& segment_ids_shape, const T* out_diff,
                                 const K* segment_ids, T* in_diff);
};

template<typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kGPU, T, K>::SegmentSumForward(
    DeviceCtx* ctx, const Shape& data_shape, const T* in, const K* segment_ids,
    const int32_t unique_ids_num, T* out) {
  const int32_t input_total_size = data_shape.elem_cnt();
  const int32_t input_outer_dim_size = data_shape.At(0);
  const int32_t inner_dim_size = input_total_size / input_outer_dim_size;
  const auto output_rows = unique_ids_num;
  const int32_t output_total_size = output_rows * inner_dim_size;
  const auto config = GetCudaLaunchConfig(output_total_size);
  auto div_up = [](int a, int b) { return (a + b - 1) / b; };
  const auto total_stripe_count = inner_dim_size * div_up(input_outer_dim_size, 8);

  SegmentSumKernel<T, K, 8><<<config.block_count, config.threads_per_block>>>(
      input_outer_dim_size, inner_dim_size, output_rows, total_stripe_count, in, segment_ids, out);
}

template<typename T, typename K>
void SegmentKernelUtilImpl<DeviceType::kGPU, T, K>::SegmentSumBackward(
    DeviceCtx* ctx, const Shape& out_diff_shape, const Shape& segment_ids_shape, const T* out_diff,
    const K* segment_ids, T* in_diff) {
  const int32_t in_diff_outer_dim_size = segment_ids_shape.At(0);
  const int32_t out_diff_total_size = out_diff_shape.elem_cnt();
  const int32_t out_diff_outer_dim_size = out_diff_shape.At(0);
  const int32_t inner_dim_size = out_diff_total_size / out_diff_outer_dim_size;
  const int32_t in_diff_total_size = inner_dim_size * in_diff_outer_dim_size;

  const auto config = GetCudaLaunchConfig(in_diff_total_size);
  auto div_up = [](int a, int b) { return (a + b - 1) / b; };
  const auto total_stripe_count = inner_dim_size * div_up(in_diff_outer_dim_size, 8);
  SegmentSumGradKernel<T, K, 8><<<config.block_count, config.threads_per_block>>>(
      in_diff_outer_dim_size, inner_dim_size, total_stripe_count, out_diff, segment_ids, in_diff);
}

// instantiate template declaration
template struct SegmentKernelUtilImpl<DeviceType::kGPU, float, int32_t>;

}  // namespace oneflow
