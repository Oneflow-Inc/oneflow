#include "oneflow/core/kernel/identify_non_small_boxes_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void IdentifyNonSmallBoxesGpu(const T* in_ptr, const int32_t num_boxes,
                                         const float min_size, int8_t* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    if (in_ptr[i * 4 + 2] - in_ptr[i * 4] >= min_size
        && in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] >= min_size) {
      out_ptr[i] = 1;
    }
  }
}

}  // namespace

template<typename T>
struct IdentifyNonSmallBoxesUtil<DeviceType::kGPU, T> {
  static void IdentifyNonSmallBoxes(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                                    const float min_size, int8_t* out_ptr) {
    IdentifyNonSmallBoxesGpu<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0,
                               ctx->cuda_stream()>>>(in_ptr, num_boxes, min_size, out_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct IdentifyNonSmallBoxesUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
