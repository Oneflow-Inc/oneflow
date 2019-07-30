#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_blob = BnInOp2Blob(ModelName());
  Blob* out_blob = BnInOp2Blob("out");
  if ((this->op_conf().trainable() && *tick_ % this->job_desc().NumOfPiecesInBatch() == 0)
      || (this->op_conf().trainable() == false && *tick_ == 0)) {
    if (this->kernel_conf().variable_conf().is_fw_inplace()) {
      CHECK_EQ(out_blob->dptr(), model_blob->dptr());
    } else {
      CHECK_NE(out_blob->dptr(), model_blob->dptr());
      out_blob->CopyDataContentFrom(ctx.device_ctx, model_blob);
    }
  } else {
    // do nothing
  }
  ++(*tick_);
}

template<DeviceType device_type, typename T>
const PbMessage& VariableKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().variable_conf();
}

template<DeviceType device_type, typename T>
const std::string& VariableKernel<device_type, T>::ModelName() const {
  return this->op_conf().variable_conf().model_name();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVariableConf, VariableKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
