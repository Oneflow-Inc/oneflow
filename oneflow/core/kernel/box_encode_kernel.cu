#include "oneflow/core/kernel/box_encode_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void EncodeGpu(const int32_t num_boxes, const T* ref_boxes_ptr, const T* boxes_ptr,
                          const float weight_x, const float weight_y, const float weight_w,
                          const float weight_h, T* boxes_delta_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    const T ref_box_x1 = ref_boxes_ptr[i * 4];
    const T ref_box_y1 = ref_boxes_ptr[i * 4 + 1];
    const T ref_box_width = ref_boxes_ptr[i * 4 + 2] - ref_box_x1 + TO_REMOVE;
    const T ref_box_height = ref_boxes_ptr[i * 4 + 3] - ref_box_y1 + TO_REMOVE;
    const T box_x1 = boxes_ptr[i * 4];
    const T box_y1 = boxes_ptr[i * 4 + 1];
    const T box_width = boxes_ptr[i * 4 + 2] - box_x1 + TO_REMOVE;
    const T box_height = boxes_ptr[i * 4 + 3] - box_y1 + TO_REMOVE;
    boxes_delta_ptr[i * 4] =
        weight_x * ((ref_box_x1 + 0.5 * ref_box_width) - (box_x1 + 0.5 * box_width)) / box_width;
    boxes_delta_ptr[i * 4 + 1] =
        weight_y * ((ref_box_y1 + 0.5 * ref_box_height) - (box_y1 + 0.5 * box_height)) / box_height;
    boxes_delta_ptr[i * 4 + 2] = weight_w * log(ref_box_width / box_width);
    boxes_delta_ptr[i * 4 + 3] = weight_h * log(ref_box_height / box_height);
  }
}

}  // namespace

template<typename T>
struct BoxEncodeUtil<DeviceType::kGPU, T> {
  static void Encode(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                     const T* boxes_ptr, const float weight_x, const float weight_y,
                     const float weight_w, const float weight_h, T* boxes_delta_ptr) {
    EncodeGpu<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        num_boxes, ref_boxes_ptr, boxes_ptr, weight_x, weight_y, weight_w, weight_h,
        boxes_delta_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct BoxEncodeUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow