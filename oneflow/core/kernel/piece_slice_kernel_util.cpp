#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PieceSliceKernelUtil<device_type>::PieceSlice(DeviceCtx* ctx, const size_t ins_idx,
                                                   const size_t valid_ins_num, const Blob* in_blob,
                                                   Blob* out_blob) {
  const size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  const char* src = in_blob->dptr<char>() + out_byte_size * ins_idx;
  char* dst = out_blob->mut_dptr<char>();
  Memcpy<device_type>(ctx, dst, src, out_byte_size);
}

template<DeviceType device_type>
void PieceSliceKernelUtil<device_type>::InstanceStack(DeviceCtx* ctx, const size_t ins_idx,
                                                      const size_t valid_ins_num,
                                                      const Blob* in_blob, Blob* out_blob) {
  const size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  const char* src = in_blob->dptr<char>();
  char* dst = out_blob->mut_dptr<char>() + in_byte_size * ins_idx;
  Memcpy<device_type>(ctx, dst, src, in_byte_size);
}

template class PieceSliceKernelUtil<DeviceType::kCPU>;
template class PieceSliceKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow