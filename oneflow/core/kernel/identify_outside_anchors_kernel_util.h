#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct IdentifyOutsideAnchorsUtil final {
  static void IdentifyOutsideAnchors(DeviceCtx* ctx, const Blob* anchors_blob,
                                     const Blob* image_size_blob, Blob* identification_blob,
                                     float tolerance);
};

}  // namespace oneflow