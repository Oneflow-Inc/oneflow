#include "oneflow/core/kernel/vstack_kernel.h"
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
    int64_t instance_num = in_blob->shape().At(0);
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(out_instance_offset),
                        in_blob->dptr<T>(), instance_num * elem_cnt_per_instance * sizeof(T));
    out_instance_offset += instance_num;
  }
  CHECK_LE(out_instance_offset, out_blob->static_shape().At(0));
}

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t total_instance_num = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    total_instance_num += BnInOp2Blob(input_bn)->dim0_valid_num(0);
  }
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_LE(total_instance_num, out_blob->dim0_inner_shape().Count(1));
  out_blob->set_dim0_valid_num(0, total_instance_num);
}

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const int64_t elem_cnt_per_instance = out_diff_blob->shape().Count(1);
  int64_t out_instance_offset = 0;
  for (const auto& in_bn : this->op_attribute().input_bns()) {
    int64_t instance_num = BnInOp2Blob(in_bn)->shape().At(0);
    Blob* in_diff_blob = BnInOp2Blob(GenDiffBn(in_bn));
    Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(),
                        out_diff_blob->dptr<T>(out_instance_offset),
                        instance_num * elem_cnt_per_instance * sizeof(T));
    out_instance_offset += instance_num;
  }
}

template<DeviceType device_type, typename T>
void VStackKernel<device_type, T>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const auto& in_bn : this->op_attribute().input_bns()) {
    BnInOp2Blob(GenDiffBn(in_bn))->set_dim0_valid_num(0, BnInOp2Blob(in_bn)->dim0_valid_num(0));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVstackConf, VStackKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
