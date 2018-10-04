#include "oneflow/core/kernel/pack_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;

  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  CHECK_EQ(total_pack_num, RoundUp(out_byte_size, in_byte_size));

  const char* src_dptr = in_blob->dptr<char>();
  char* dst_dptr = out_blob->mut_dptr<char>() + in_byte_size * in_index;
  size_t actual_copy_byte_size = in_byte_size;
  if (in_index + 1 == total_pack_num) {
    actual_copy_byte_size = out_byte_size - in_byte_size * in_index;
  }
  Memcpy<device_type>(ctx.device_ctx, dst_dptr, src_dptr, actual_copy_byte_size);
}

}  // namespace oneflow
