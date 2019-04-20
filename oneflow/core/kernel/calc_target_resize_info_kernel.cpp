#include "oneflow/core/kernel/calc_target_resize_info_kernel.h"

namespace oneflow {

template<typename T>
void CalcTargetResizeInfoKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const T* in_ptr = in_blob->dptr<T>();
  float* scale_ptr = BnInOp2Blob("scale")->mut_dptr<float>();
  T* out_ptr = BnInOp2Blob("out")->mut_dptr<T>();
  const auto& conf = this->op_conf().calc_target_resize_info_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();

  FOR_RANGE(T, i, 0, in_blob->shape().At(0)) {
    const int32_t origin_image_height = in_ptr[i * 2];
    const int32_t origin_image_width = in_ptr[i * 2 + 1];
    const T im_size_min = std::min(origin_image_height, origin_image_width);
    const T im_size_max = std::max(origin_image_height, origin_image_width);
    scale_ptr[i] = static_cast<float>(target_size) / static_cast<float>(im_size_min);
    if (std::round(scale_ptr[i] * im_size_max) > max_size) {
      scale_ptr[i] = static_cast<float>(max_size) / static_cast<float>(im_size_max);
    }
    out_ptr[2 * i] = static_cast<T>(origin_image_height * scale_ptr[i]);
    out_ptr[2 * i + 1] = static_cast<T>(origin_image_width * scale_ptr[i]);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalcTargetResizeInfoConf, CalcTargetResizeInfoKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
