#include "oneflow/core/kernel/calc_target_resize_info_kernel.h"

namespace oneflow {

template<typename T>
void CalcTargetResizeInfoKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* origin_height_blob = BnInOp2Blob("origin_height");
  const T* origin_height = origin_height_blob->dptr<T>();
  const T* origin_width = BnInOp2Blob("origin_width")->dptr<T>();
  float* scale = BnInOp2Blob("scale")->mut_dptr<float>();
  T* resized_image_size = BnInOp2Blob("resized_image_size")->mut_dptr<T>();
  const auto& conf = this->op_conf().calc_target_resize_info_conf();
  const T target_size = conf.target_size();
  const T max_size = conf.max_size();

  FOR_RANGE(T, i, 0, origin_height_blob->shape().elem_cnt()) {
    const T im_size_min = std::min(origin_height[i], origin_width[i]);
    const T im_size_max = std::max(origin_height[i], origin_width[i]);
    scale[i] = static_cast<float>(target_size) / static_cast<float>(im_size_min);
    if (std::round(scale[i] * im_size_max) > max_size) {
      scale[i] = static_cast<float>(max_size) / static_cast<float>(im_size_max);
    }
    resized_image_size[2 * i] = static_cast<T>(origin_height[i] * scale[i]);
    resized_image_size[2 * i + 1] = static_cast<T>(origin_width[i] * scale[i]);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalcTargetResizeInfoConf, CalcTargetResizeInfoKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
