#include "oneflow/core/kernel/unpack_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void UnpackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t out_index = res->first;
  size_t total_unpack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  CHECK_EQ(total_unpack_num, RoundUp(in_byte_size, out_byte_size));

  const char* src_dptr = in_blob->dptr<char>() + out_byte_size * out_index;
  char* dst_dptr = out_blob->dptr<char>();
  size_t actual_copy_byte_size = out_byte_size;
  if (out_index + 1 == total_unpack_num) {
    actual_copy_byte_size = in_byte_size - out_byte_size * out_index;
    size_t noncopy_byte_size = out_byte_size * total_unpack_num - in_byte_size;
    Memset<device_type>(ctx.device_ctx, dst_dptr + actual_copy_byte_size, 0, noncopy_byte_size);
  }
  Memcpy<device_type>(ctx.device_ctx, dst_dptr, src_dptr, actual_copy_byte_size);
}

}  // namespace oneflow
