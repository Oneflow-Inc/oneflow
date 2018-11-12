#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  if ((this->op_conf().trainable() && *tick_ % Global<JobDesc>::Get()->NumOfPiecesInBatch() == 0)
      || (this->op_conf().trainable() == false && *tick_ == 0)) {
    out_blob->CopyDataContentFrom(ctx.device_ctx, weight_blob);
  } else {
    // do nothing
  }
  ++(*tick_);
}

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->op_conf().trainable());
  BnInOp2Blob("weight_diff")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
}

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, GetMsgPtrFromPbMessage(this->op_conf().variable_conf(), "initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));
}

template<DeviceType device_type, typename T>
void VariableKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, weight_blob,
                                                "weight", weight_blob->shape().At(0),
                                                weight_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
const PbMessage& VariableKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().variable_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVariableConf, VariableKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
