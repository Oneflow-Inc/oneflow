#ifndef ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceKernelUtil final {
 public:
  static void PieceSlice(DeviceCtx* ctx, const size_t ins_idx, const size_t valid_ins_num,
                         const Blob* in_blob, Blob* out_blob);
  static void InstanceStack(DeviceCtx* ctx, const size_t ins_idx, const size_t valid_ins_num,
                            const Blob* in_blob, Blob* out_blob);
  static void SliceInstanceShape(const Blob* in_blob, Blob* out_blob);
  static bool StackInstanceShape(const bool is_first_instance, const Blob* in_blob, Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_UTIL_H_