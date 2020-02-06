#include "oneflow/core/kernel/calc_iou_matrix_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CalIoUMatrixKernel(const T* boxes1_ptr, const int32_t num_boxes1,
                                   const T* boxes2_ptr, const int32_t num_boxes2,
                                   float* iou_matrix_ptr) {
  const T TO_REMOVE = 1.0;
  for (size_t i = blockIdx.x; i < num_boxes1; i += gridDim.x) {
    for (size_t j = threadIdx.x; j < num_boxes2; j += blockDim.x) {
      const T box1_x1 = boxes1_ptr[i * 4];
      const T box1_y1 = boxes1_ptr[i * 4 + 1];
      const T box1_x2 = boxes1_ptr[i * 4 + 2];
      const T box1_y2 = boxes1_ptr[i * 4 + 3];
      const T box2_x1 = boxes2_ptr[j * 4];
      const T box2_y1 = boxes2_ptr[j * 4 + 1];
      const T box2_x2 = boxes2_ptr[j * 4 + 2];
      const T box2_y2 = boxes2_ptr[j * 4 + 3];
      const T iw = min(box1_x2, box2_x2) - max(box1_x1, box2_x1) + TO_REMOVE;
      const T ih = min(box1_y2, box2_y2) - max(box1_y1, box2_y1) + TO_REMOVE;
      if (iw <= 0.0 || ih <= 0.0) {
        iou_matrix_ptr[i * num_boxes2 + j] = 0.0;
      } else {
        const T area_inner = iw * ih;
        const T area_box1 = (box1_y2 - box1_y1 + TO_REMOVE) * (box1_x2 - box1_x1 + TO_REMOVE);
        const T area_box2 = (box2_y2 - box2_y1 + TO_REMOVE) * (box2_x2 - box2_x1 + TO_REMOVE);
        iou_matrix_ptr[i * num_boxes2 + j] = area_inner / (area_box1 + area_box2 - area_inner);
      }
    }
  }
}

}  // namespace

template<typename T>
struct CalcIoUMatrixUtil<DeviceType::kGPU, T> {
  static void CalcIoUMatrix(DeviceCtx* ctx, const T* boxes1_ptr, const int32_t num_boxes1,
                            const T* boxes2_ptr, const int32_t num_boxes2, float* iou_matrix_ptr) {
    CalIoUMatrixKernel<<<std::min(num_boxes1, kCudaMaxBlocksNum),
                         std::min(num_boxes2, kCudaThreadsNumPerBlock), 0, ctx->cuda_stream()>>>(
        boxes1_ptr, num_boxes1, boxes2_ptr, num_boxes2, iou_matrix_ptr);
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct CalcIoUMatrixUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow