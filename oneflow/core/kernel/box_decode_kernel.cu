#include "oneflow/core/kernel/box_decode_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuForward(const int32_t num_boxes, const T* ref_boxes_ptr,
                           const T* boxes_delta_ptr, const float weight_x, const float weight_y,
                           const float weight_w, const float weight_h, T* boxes_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(i, num_boxes) {
    const T ref_box_x1 = ref_boxes_ptr[i * 4];
    const T ref_box_y1 = ref_boxes_ptr[i * 4 + 1];
    const T ref_box_width = ref_boxes_ptr[i * 4 + 2] - ref_box_x1 + TO_REMOVE;
    const T ref_box_height = ref_boxes_ptr[i * 4 + 3] - ref_box_y1 + TO_REMOVE;
    const T box_ctr_x =
        boxes_delta_ptr[i * 4] / weight_x * ref_box_width + ref_box_x1 + 0.5 * ref_box_width;
    const T box_ctr_y =
        boxes_delta_ptr[i * 4 + 1] / weight_y * ref_box_height + ref_box_y1 + 0.5 * ref_box_height;
    const T box_width = exp(boxes_delta_ptr[i * 4 + 2] / weight_w) * ref_box_width;
    const T box_height = exp(boxes_delta_ptr[i * 4 + 3] / weight_h) * ref_box_height;
    boxes_ptr[i * 4] = box_ctr_x - 0.5 * box_width;
    boxes_ptr[i * 4 + 1] = box_ctr_y - 0.5 * box_height;
    boxes_ptr[i * 4 + 2] = box_ctr_x + 0.5 * box_width - 1;
    boxes_ptr[i * 4 + 3] = box_ctr_y + 0.5 * box_height - 1;
  }
}

}  // namespace

template<typename T>
struct BoxDecodeUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                      const T* boxes_delta_ptr, const float weight_x, const float weight_y,
                      const float weight_w, const float weight_h, T* boxes_ptr) {
    GpuForward<<<BlocksNum4ThreadsNum(num_boxes), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        num_boxes, ref_boxes_ptr, boxes_delta_ptr, weight_x, weight_y, weight_w, weight_h,
        boxes_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct BoxDecodeUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
