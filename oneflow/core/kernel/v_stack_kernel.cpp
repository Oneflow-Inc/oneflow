#include "oneflow/core/kernel/v_stack_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t elem_cnt_per_instance = out_blob->shape().Count(1);
  int64_t out_instance_offset = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    const Blob* in_blob = BnInOp2Blob(input_bn);
    int64_t instance_num = in_blob->available_instance_num();
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(out_instance_offset),
                        in_blob->dptr<T>(), instance_num * elem_cnt_per_instance * sizeof(T));
    out_instance_offset += instance_num;
  }
  CHECK_LE(out_instance_offset, out_blob->shape().At(0));
}

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::ForwardVaryingInstanceNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t total_instance_num = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    total_instance_num += BnInOp2Blob(input_bn)->varying_instance_num(0);
  }
  BnInOp2Blob("out")->set_varying_instance_num(0, total_instance_num);
}

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVStackConf, VStackKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
