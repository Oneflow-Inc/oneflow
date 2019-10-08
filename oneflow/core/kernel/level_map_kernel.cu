#include "oneflow/core/kernel/level_map_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int64_t num_boxes, const T* in_ptr, const int32_t canonical_level,
                           const int32_t canonical_scale, const int32_t min_level,
                           const int32_t max_level, const float epsilon, int32_t* out_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    const T scale = sqrt((in_ptr[i * 4 + 2] - in_ptr[i * 4] + TO_REMOVE)
                         * (in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] + TO_REMOVE));
    const int32_t target_level = floor(canonical_level + log2(scale / canonical_scale + epsilon));
    out_ptr[i] = min(max(target_level, min_level), max_level) - min_level;
  }
}

}  // namespace

template<typename T>
struct LevelMapUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t num_boxes, const T* in_ptr,
                      const int32_t canonical_level, const float canonical_scale,
                      const int32_t min_level, const int32_t max_level, const float epsilon,
                      int32_t* out_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        num_boxes, in_ptr, canonical_level, canonical_scale, min_level, max_level, epsilon,
        out_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct LevelMapUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
