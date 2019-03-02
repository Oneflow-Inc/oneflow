#include "oneflow/core/kernel/calculate_scale_kernel.h"

namespace oneflow {

template<typename T>
void CalculateScaleKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* height_blob = BnInOp2Blob("height");
  const int32_t* height = height_blob->dptr<int32_t>();
  const int32_t* weight = BnInOp2Blob("weight")->dptr<int32_t>();
  T* scale = BnInOp2Blob("scale")->mut_dptr<T>();
  const auto& conf = this->op_conf().calculate_scale_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();

  FOR_RANGE(int32_t, i, 0, height_blob->shape().elem_cnt()) {
    const int32_t im_size_min = std::min(height[i], weight[i]);
    const int32_t im_size_max = std::max(height[i], weight[i]);
    T im_scale = static_cast<T>(target_size) / static_cast<T>(im_size_min);
    if (std::round(im_scale * im_size_max) > max_size) {
      im_scale = static_cast<T>(max_size) / static_cast<T>(im_size_max);
    }
    scale[i] = im_scale;
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalculateScaleConf, CalculateScaleKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
