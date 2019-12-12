#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct CalcIoUMatrixUtil {
  static void CalcIoUMatrix(DeviceCtx* ctx, const T* boxes1_ptr, const int32_t num_boxes1,
                            const T* boxes2_ptr, const int32_t num_boxes2, float* iou_matrix_ptr);
};

}  // namespace oneflow