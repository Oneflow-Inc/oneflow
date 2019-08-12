#include "oneflow/core/kernel/convert_box_mode_kernel.h"

namespace oneflow {

template<typename T>
struct ConvertBoxModeUtil<DeviceType::kGPU, T> {
  static void Forward() {
    // TODO
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct ConvertBoxModeUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
