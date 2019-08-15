#ifndef ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceV2KernelUtil final {
 public:
  static void PieceSlice(DeviceCtx* ctx, const Blob* in_blob, std::vector<Blob*>& out_blobs);
  static void InstanceStack(DeviceCtx* ctx, const std::vector<const Blob*>& in_blobs,
                            Blob* out_blob);
  static void SliceInstanceShape(const Blob* in_blob, std::vector<Blob*>& out_blobs);
  static void StackInstanceShape(const std::vector<const Blob*>& in_blobs, Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PIECE_SLICE_V2_KERNEL_UTIL_H_