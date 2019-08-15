#include "oneflow/core/kernel/non_maximum_supression_kernel.h"

namespace oneflow {

template<typename T>
struct NonMaximumSupressionUtil<DeviceType::kGPU, T> {
  static void Forward() {
    // TODO
  }
};

#define MAKE_ENTRY(type_cpp, type_proto) \
  template struct NonMaximumSupressionUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
