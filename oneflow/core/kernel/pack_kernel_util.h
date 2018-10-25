#ifndef ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class PackKernelUtil final {
 public:
  static void Pack(DeviceCtx* ctx, size_t in_index, size_t total_pack_num, const Blob* in_blob,
                   Blob* out_blob);
  static void Unpack(DeviceCtx* ctx, size_t out_index, size_t total_unpack_num, const Blob* in_blob,
                     Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_
