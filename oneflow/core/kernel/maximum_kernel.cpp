#include "oneflow/core/kernel/maximum_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void MaximumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MaximumOpConf& maximum_conf = this->op_conf().maximum_conf();
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob0 = BnInOp2Blob("in_0");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(),
                      in_blob0->dptr<T>(),
                      out_blob->ByteSizeOfDataContentField());
  Blob* mask_blob = BnInOp2Blob("mask");
  // set mask to zero because out is set to in0
  Memset<device_type>(ctx.device_ctx, mask_blob->mut_dptr(), 0,
                      mask_blob->ByteSizeOfDataContentField());
  const int count = out_blob->shape().elem_cnt();
  // out = max(in0, in1, in2, in3....)
  // mask = index_of_max(in0, in1, in2, in3....)
  // start at index of 1 because out is set to in0
  for (int i = 1; i < maximum_conf.in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    const Blob* in_blob = BnInOp2Blob(ibn);
    for (int idx = 0; idx < count; ++idx) {
      if (in_blob->dptr<T>()[idx] > out_blob->mut_dptr<T>()[idx]) {
        out_blob->mut_dptr<T>()[idx] = in_blob->dptr<T>()[idx];
        mask_blob->mut_dptr<T>()[idx] = i;
      }
    }
  }
}

template<DeviceType device_type, typename T>
void MaximumKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* mask_blob = BnInOp2Blob("mask");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const MaximumOpConf& maximum_conf = this->op_conf().maximum_conf();
  const int count = mask_blob->shape().elem_cnt();
  // in_diff = out_diff if it is the max one
  for (int i = 0; i < maximum_conf.in_size(); ++i) {
    for (int idx = 0; idx < count; ++idx) {
      std::string idbn = GenDiffBn("in_" + std::to_string(i));
      Blob* in_diff_blob = BnInOp2Blob(idbn);
      Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                          in_diff_blob->ByteSizeOfDataContentField());
      if (i == mask_blob->dptr<T>()[idx]) {
        in_diff_blob->mut_dptr<T>()[idx] = out_diff_blob->dptr<T>()[idx];
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaximumConf, MaximumKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
