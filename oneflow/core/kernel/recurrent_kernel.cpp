#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
bool RecurrentKernel<device_type, T>::Ish0Model() const {
  auto& input_bns = this->kernel_conf().input_bns();
  return find(input_bns.begin(), input_bns.end(), "h0") == input_bns.end();
}

template<DeviceType device_type, typename T>
Blob* RecurrentKernel<device_type, T>::GetHiddenBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
  return BnInOp2Blob("ht_1");
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->Ish0Model()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().recurrent_conf(),
                          init_hidden_initializer),
        random_seed_gen(), BnInOp2Blob("h0"));
  }
  VirtualInitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (Ish0Model()) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("h0"),
        "h0", BnInOp2Blob("h0")->shape().At(0), 1);
  }
  VirtualInitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir,
                               BnInOp2Blob);
}

namespace {

#define RECURRENT_KERNEL_PAIR_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(BasicRnnKernel, RecurrentOpConf::kBasicRnnCell)

#define MAKE_RECURRENT_KERNEL_CREATOR_ENTRY(recurrnt_kernel_pair, device_type, \
                                            data_type_pair)                    \
  {GetHashKey(OF_PP_PAIR_SECOND(recurrnt_kernel_pair), device_type,            \
              OF_PP_PAIR_SECOND(data_type_pair)),                              \
   []() {                                                                      \
     return new OF_PP_PAIR_FIRST(                                              \
         recurrnt_kernel_pair)<device_type,                                    \
                               OF_PP_PAIR_FIRST(data_type_pair)>();            \
   }},

Kernel* CreateKernel(DeviceType dev_type, const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          MAKE_RECURRENT_KERNEL_CREATOR_ENTRY, RECURRENT_KERNEL_PAIR_SEQ,
          DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(
      dev_type, kernel_conf.op_conf().recurrent_conf().rnn_type_case(),
      kernel_conf.data_type()))();
}

COMMAND(AddKernelCreator(OperatorConf::kRecurrentConf, CreateKernel));

}  // namespace

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class RecurrentKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
