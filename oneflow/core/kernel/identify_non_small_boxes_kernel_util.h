#ifndef ONEFLOW_CORE_KERNEL_IDENTIFY_NON_SMALL_BOXES_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_IDENTIFY_NON_SMALL_BOXES_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct IdentifyNonSmallBoxesUtil {
  static void IdentifyNonSmallBoxes(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                                    const float min_size, int8_t* out_ptr);
};

}  // namespace oneflow

#endif
