#include "oneflow/core/kernel/layer_normalization_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::ForwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {
  // TODO
}

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::BackwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {
  // TODO
}

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx*, std::mt19937*, std::function<Blob*(const std::string&)>) const {
  // TODO
}

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx*, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLayerNormConf, LayerNormKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
