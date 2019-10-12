#ifndef ONEFLOW_CORE_KERNEL_BOX_ENCODE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BOX_ENCODE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct BoxEncodeUtil {
  static void Encode(DeviceCtx* ctx, const int32_t num_boxes, const T* ref_boxes_ptr,
                     const T* boxes_ptr, const float weight_x, const float weight_y,
                     const float weight_w, const float weight_h, T* boxes_delta_ptr);
};

}  // namespace oneflow

#endif
