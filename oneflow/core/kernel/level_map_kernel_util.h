#ifndef ONEFLOW_CORE_KERNEL_LEVEL_MAP_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_LEVEL_MAP_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct LevelMapUtil {
  static void Forward(DeviceCtx* ctx, const int64_t num_boxes, const T* in_ptr,
                      const int32_t canonical_level, const float canonical_scale,
                      const int32_t min_level, const int32_t max_level, const float epsilon,
                      int32_t* out_ptr);
};

}  // namespace oneflow

#endif
