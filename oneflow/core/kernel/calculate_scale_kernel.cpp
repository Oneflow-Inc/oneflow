#include "oneflow/core/kernel/calculate_scale_kernel.h"

namespace oneflow {

template<typename T>
void CalculateScaleKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* origin_height_blob = BnInOp2Blob("origin_height");
  const int32_t* origin_height = origin_height_blob->dptr<int32_t>();
  const int32_t* origin_width = BnInOp2Blob("origin_width")->dptr<int32_t>();
  T* scale = BnInOp2Blob("scale")->mut_dptr<T>();
  int32_t* height = BnInOp2Blob("height")->mut_dptr<int32_t>();
  int32_t* width = BnInOp2Blob("width")->mut_dptr<int32_t>();
  const auto& conf = this->op_conf().calculate_scale_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();

  FOR_RANGE(int32_t, i, 0, origin_height_blob->shape().elem_cnt()) {
    const int32_t im_size_min = std::min(origin_height[i], origin_width[i]);
    const int32_t im_size_max = std::max(origin_height[i], origin_width[i]);
    scale[i] = static_cast<T>(target_size) / static_cast<T>(im_size_min);
    if (std::round(scale[i] * im_size_max) > max_size) {
      scale[i] = static_cast<T>(max_size) / static_cast<T>(im_size_max);
    }
    height[i] = static_cast<int32_t>(origin_height[i] * scale[i]);
    width[i] = static_cast<int32_t>(origin_width[i] * scale[i]);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalculateScaleConf, CalculateScaleKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
