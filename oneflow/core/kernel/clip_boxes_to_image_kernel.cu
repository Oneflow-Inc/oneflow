#include "oneflow/core/kernel/clip_boxes_to_image_kernel.h"

namespace oneflow {

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kGPU, T> {
  static void Forward() {
    // TODO
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ClipBoxesToImageUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)
#undef MAKE_ENTRY

}  // namespace oneflow
