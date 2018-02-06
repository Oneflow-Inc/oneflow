#include "oneflow/core/kernel/recurrent_kernel.h"
#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext*) {
  auto& input_bns = this->kernel_conf().input_bns();
  need_external_h0_ =
      std::find(input_bns.begin(), input_bns.end(), "h0") == input_bns.end();
}

template<DeviceType device_type, typename T>
bool RecurrentKernel<device_type, T>::NeedExternalH0() const {
  return need_external_h0_;
}

template<DeviceType device_type, typename T>
Blob* RecurrentKernel<device_type, T>::GetHiddenBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
  return BnInOp2Blob("rec_in");
}

template<DeviceType device_type, typename T>
Blob* RecurrentKernel<device_type, T>::GetHiddenDiffBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0_diff"); }
  return BnInOp2Blob("rec_in_diff");
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (NeedExternalH0()) {
    const PbMessage& recurrent_conf = GetRecurrentOpConf();
    const InitializerConf* init_hidden_initializer = nullptr;
    if (HasInitHiddenInitializer()) {
      init_hidden_initializer = static_cast<const InitializerConf*>(
          &GetMessageFromPbMessage(recurrent_conf, "init_hidden_initializer"));
    }
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx, init_hidden_initializer, random_seed_gen(),
        BnInOp2Blob("h0"));
  }
  VirtualInitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (NeedExternalH0()) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("h0"),
        "h0", BnInOp2Blob("h0")->shape().At(0), 1);
  }
  VirtualInitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir,
                               BnInOp2Blob);
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class RecurrentKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
