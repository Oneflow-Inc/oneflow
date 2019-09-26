#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct IdentifyNonSmallBoxesUtil {
  static void IdentifyNonSmallBoxes(DeviceCtx* ctx, const T* in_ptr, const int32_t num_boxes,
                                    const float min_size, int8_t* out_ptr);
};

}  // namespace oneflow