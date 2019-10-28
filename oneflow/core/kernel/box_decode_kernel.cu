#include "oneflow/core/kernel/box_decode_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void DecodeGpu(const int32_t num_boxes_delta, const int32_t num_ref_boxes,
                          const T* ref_boxes_ptr, const T* boxes_delta_ptr, const float weight_x,
                          const float weight_y, const float weight_w, const float weight_h,
                          T* boxes_ptr) {
  const T TO_REMOVE = 1.0;
  CUDA_1D_KERNEL_LOOP(delta_idx, num_boxes_delta) {
    const T dx = boxes_delta_ptr[delta_idx * 4] / weight_x;
    const T dy = boxes_delta_ptr[delta_idx * 4 + 1] / weight_y;
    const T dw = boxes_delta_ptr[delta_idx * 4 + 2] / weight_w;
    const T dh = boxes_delta_ptr[delta_idx * 4 + 3] / weight_h;

    const int32_t ref_box_idx = delta_idx / (num_boxes_delta / num_ref_boxes);
    const T ref_box_width =
        ref_boxes_ptr[ref_box_idx * 4 + 2] - ref_boxes_ptr[ref_box_idx * 4] + TO_REMOVE;
    const T ref_box_height =
        ref_boxes_ptr[ref_box_idx * 4 + 3] - ref_boxes_ptr[ref_box_idx * 4 + 1] + TO_REMOVE;
    const T ref_box_ctr_x = ref_boxes_ptr[ref_box_idx * 4] + 0.5 * ref_box_width;
    const T ref_box_ctr_y = ref_boxes_ptr[ref_box_idx * 4 + 1] + 0.5 * ref_box_height;

    const T pred_box_ctr_x = dx * ref_box_width + ref_box_ctr_x;
    const T pred_box_ctr_y = dy * ref_box_height + ref_box_ctr_y;
    const T pred_box_w = exp(dw) * ref_box_width;
    const T pred_box_h = exp(dh) * ref_box_height;

    boxes_ptr[delta_idx * 4] = pred_box_ctr_x - 0.5 * pred_box_w;
    boxes_ptr[delta_idx * 4 + 1] = pred_box_ctr_y - 0.5 * pred_box_h;
    boxes_ptr[delta_idx * 4 + 2] = pred_box_ctr_x + 0.5 * pred_box_w - 1.0;
    boxes_ptr[delta_idx * 4 + 3] = pred_box_ctr_y + 0.5 * pred_box_h - 1.0;
  }
}

}  // namespace

template<typename T>
struct BoxDecodeUtil<DeviceType::kGPU, T> {
  static void Decode(DeviceCtx* ctx, const int32_t num_boxes_delta, const int32_t num_ref_boxes,
                     const T* ref_boxes_ptr, const T* boxes_delta_ptr, const float weight_x,
                     const float weight_y, const float weight_w, const float weight_h,
                     T* boxes_ptr) {
    DecodeGpu<<<BlocksNum4ThreadsNum(num_boxes_delta), kCudaThreadsNumPerBlock, 0,
                ctx->cuda_stream()>>>(num_boxes_delta, num_ref_boxes, ref_boxes_ptr,
                                      boxes_delta_ptr, weight_x, weight_y, weight_w, weight_h,
                                      boxes_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) template struct BoxDecodeUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow