#include "oneflow/core/kernel/prelu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PReluKernelUtil<device_type, T>::Forward(ctx, this->op_conf().prelu_conf(), BnInOp2Blob("in"),
                                           BnInOp2Blob("weight"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  PReluKernelUtil<device_type, T>::Backward(ctx, this->op_conf().prelu_conf(), BnInOp2Blob("in"),
                                            BnInOp2Blob("weight"), BnInOp2Blob("out_diff"),
                                            in_diff_blob, BnInOp2Blob("weight_diff"));
}

template<typename T>
struct PReluKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                      const Blob* weight_blob, Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                       const Blob* weight_blob, const Blob* out_diff_blob, Blob* in_diff_blob,
                       Blob* weight_diff_blob) {
    UNIMPLEMENTED();
  }
};

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& prelu_conf = this->op_conf().prelu_conf();
  float alpha_init = prelu_conf.alpha_init();
  InitializerConf alpha_init_conf;
  alpha_init_conf.mutable_constant_conf()->set_value(alpha_init);
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &alpha_init_conf, 0,
                                                       BnInOp2Blob("weight"));
}

template<DeviceType device_type, typename T>
void PReluKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = weight_blob->shape().At(0);
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, weight_blob,
                                                "weight", dim_num, 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPreluConf, PReluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
