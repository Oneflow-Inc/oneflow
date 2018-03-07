#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AddOpConf& add_conf = this->op_conf().add_conf();
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob0 = BnInOp2Blob("in_0");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(),
                      in_blob0->dptr<T>(),
                      out_blob->ByteSizeOfDataContentField());
  const int count = out_blob->shape().elem_cnt();
  for (int i = 1; i < add_conf.in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    const Blob* in_blob = BnInOp2Blob(ibn);
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, count, 1.0f,
                                     in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const AddOpConf& add_conf = this->op_conf().add_conf();
  for (int i = 0; i < add_conf.in_size(); ++i) {
    std::string idbn = GenDiffBn("in_" + std::to_string(i));
    Blob* in_diff_blob = BnInOp2Blob(idbn);
    Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(),
                        out_diff_blob->dptr<T>(),
                        out_diff_blob->ByteSizeOfDataContentField());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddConf, AddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
