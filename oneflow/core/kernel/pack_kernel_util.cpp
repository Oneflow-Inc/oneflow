#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernelUtil<device_type>::Pack(DeviceCtx* ctx, size_t in_index, size_t total_pack_num,
                                       const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  CHECK_EQ(total_pack_num, (out_byte_size + in_byte_size - 1) / in_byte_size);

  const char* src_dptr = in_blob->dptr<char>();
  char* dst_dptr = out_blob->mut_dptr<char>() + in_byte_size * in_index;
  size_t actual_copy_byte_size = in_byte_size;
  if (in_index + 1 == total_pack_num) {
    actual_copy_byte_size = out_byte_size - in_byte_size * in_index;
  }
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, actual_copy_byte_size);
}

template<DeviceType device_type>
void PackKernelUtil<device_type>::Unpack(DeviceCtx* ctx, size_t out_index, size_t total_unpack_num,
                                         const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  CHECK_EQ(total_unpack_num, (in_byte_size + out_byte_size - 1) / out_byte_size);

  const char* src_dptr = in_blob->dptr<char>() + out_byte_size * out_index;
  char* dst_dptr = out_blob->mut_dptr<char>();
  size_t actual_copy_byte_size = out_byte_size;
  if (out_index + 1 == total_unpack_num) {
    actual_copy_byte_size = in_byte_size - out_byte_size * out_index;
    size_t noncopy_byte_size = out_byte_size * total_unpack_num - in_byte_size;
    Memset<device_type>(ctx, dst_dptr + actual_copy_byte_size, 0, noncopy_byte_size);
  }
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, actual_copy_byte_size);
}

template class PackKernelUtil<DeviceType::kCPU>;
template class PackKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow
