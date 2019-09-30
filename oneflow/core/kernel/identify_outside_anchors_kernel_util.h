#ifndef ONEFLOW_CORE_KERNEL_IDENTIFY_OUTSIDE_ANCHORS_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_IDENTIFY_OUTSIDE_ANCHORS_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct IdentifyOutsideAnchorsUtil final {
  static void IdentifyOutsideAnchors(DeviceCtx* ctx, const Blob* anchors_blob,
                                     const Blob* image_size_blob, Blob* identification_blob,
                                     float tolerance);
};

}  // namespace oneflow

#endif
