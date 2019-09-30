#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernelUtil<device_type>::Pack(DeviceCtx* ctx, size_t in_index, size_t total_pack_num,
                                       const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfBlobBody();
  size_t out_byte_size = out_blob->ByteSizeOfBlobBody();
  CHECK_EQ(total_pack_num, out_byte_size / in_byte_size);

  const char* src_dptr = in_blob->dptr<char>();
  char* dst_dptr = out_blob->mut_dptr<char>() + in_byte_size * in_index;
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, in_byte_size);
}

template<DeviceType device_type>
void PackKernelUtil<device_type>::Unpack(DeviceCtx* ctx, size_t out_index, size_t total_unpack_num,
                                         const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfBlobBody();
  size_t out_byte_size = out_blob->ByteSizeOfBlobBody();
  CHECK_EQ(total_unpack_num, in_byte_size / out_byte_size);

  const char* src_dptr = in_blob->dptr<char>() + out_byte_size * out_index;
  char* dst_dptr = out_blob->mut_dptr<char>();
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, out_byte_size);
}

template class PackKernelUtil<DeviceType::kCPU>;
template class PackKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow
