#include "oneflow/core/kernel/maskrcnn_concat_uncontiguous_dim1_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MaskrcnnConcatUncontiguousDim1Kernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  int32_t dim1_valid_num_acc = 0;
  FOR_RANGE(int32_t, i, 0, in_blob->shape().At(0)) {
    dim1_valid_num_acc += in_blob->dim1_valid_num(i);
  }
  BnInOp2Blob("out")->set_dim0_valid_num(0, dim1_valid_num_acc);
}

template<DeviceType device_type, typename T>
void MaskrcnnConcatUncontiguousDim1Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  char* src = in_blob->mut_dptr<char>();
  char* dst = BnInOp2Blob("out")->mut_dptr<char>();
  FOR_RANGE(int32_t, i, 0, in_blob->shape().At(0)) {
    size_t instance_byte_size = in_blob->dim1_valid_num(i) * in_blob->shape().Count(2) * sizeof(T);
    Memcpy<device_type>(ctx.device_ctx, dst, src, instance_byte_size);
    src += in_blob->shape().Count(1) * sizeof(T);
    dst += instance_byte_size;
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaskrcnnConcatUncontiguousDim1Conf,
                           MaskrcnnConcatUncontiguousDim1Kernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
