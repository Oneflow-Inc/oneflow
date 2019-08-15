#include "oneflow/core/kernel/clip_boxes_to_image_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ClipBoxesToImageGpu(const int32_t num_boxes, const T* boxes_ptr,
                                    int32_t image_height, int32_t image_width, T* out_ptr) {
  const T TO_REMOVE = 1;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    out_ptr[i * 4] =
        min(max(boxes_ptr[i * 4], ZeroVal<T>::value), static_cast<T>(image_width) - TO_REMOVE);
    out_ptr[i * 4 + 1] =
        min(max(boxes_ptr[i * 4 + 1], ZeroVal<T>::value), static_cast<T>(image_height) - TO_REMOVE);
    out_ptr[i * 4 + 2] =
        min(max(boxes_ptr[i * 4 + 2], ZeroVal<T>::value), static_cast<T>(image_width) - TO_REMOVE);
    out_ptr[i * 4 + 3] =
        min(max(boxes_ptr[i * 4 + 3], ZeroVal<T>::value), static_cast<T>(image_height) - TO_REMOVE);
  }
}

}  // namespace

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* boxes_ptr,
                      const int32_t* image_size_ptr, T* out_ptr) {
    const int32_t image_height = image_size_ptr[0];
    const int32_t image_width = image_size_ptr[1];
    ClipBoxesToImageGpu<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0,
                          ctx->cuda_stream()>>>(num_boxes, boxes_ptr, image_height, image_width,
                                                out_ptr);
  };
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ClipBoxesToImageUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)
#undef MAKE_ENTRY

}  // namespace oneflow
