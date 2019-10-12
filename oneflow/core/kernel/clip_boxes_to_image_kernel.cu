#include "oneflow/core/kernel/clip_boxes_to_image_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ClipBoxesGpu(const int32_t num_boxes, const T* boxes_ptr,
                             const int32_t* image_size_ptr, T* out_ptr) {
  const int32_t image_height = image_size_ptr[0];
  const int32_t image_width = image_size_ptr[1];
  const T TO_REMOVE = 1;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    out_ptr[i * 4] =
        min(max(boxes_ptr[i * 4], GetZeroVal<T>()), static_cast<T>(image_width) - TO_REMOVE);
    out_ptr[i * 4 + 1] =
        min(max(boxes_ptr[i * 4 + 1], GetZeroVal<T>()), static_cast<T>(image_height) - TO_REMOVE);
    out_ptr[i * 4 + 2] =
        min(max(boxes_ptr[i * 4 + 2], GetZeroVal<T>()), static_cast<T>(image_width) - TO_REMOVE);
    out_ptr[i * 4 + 3] =
        min(max(boxes_ptr[i * 4 + 3], GetZeroVal<T>()), static_cast<T>(image_height) - TO_REMOVE);
  }
}

}  // namespace

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kGPU, T> {
  static void ClipBoxes(DeviceCtx* ctx, const int32_t num_boxes, const T* boxes_ptr,
                        const int32_t* image_size_ptr, T* out_ptr) {
    ClipBoxesGpu<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0,
                   ctx->cuda_stream()>>>(num_boxes, boxes_ptr, image_size_ptr, out_ptr);
  };
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ClipBoxesToImageUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)
#undef MAKE_ENTRY

}  // namespace oneflow