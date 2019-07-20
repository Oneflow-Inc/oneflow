#include "oneflow/core/kernel/bitonic_sort_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BitonicSortKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  CHECK_EQ(in_byte_size, out_byte_size);
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                      in_byte_size);
  BitonicSortUtil<device_type, T>::Forward(ctx.device_ctx, in_blob->shape().At(0),
                                           in_blob->shape().At(1), out_blob->mut_dptr<T>());
}

template<typename T>
struct BitonicSortUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t instance_num, const int32_t instance_size,
                      T* out) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBitonicSortConf, BitonicSortKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
