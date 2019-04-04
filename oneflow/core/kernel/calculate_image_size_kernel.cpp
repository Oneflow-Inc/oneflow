#include "oneflow/core/kernel/calculate_image_size_kernel.h"

namespace oneflow {

template<typename T>
void CalculateImageSizeKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* origin_height_blob = BnInOp2Blob("origin_height");
  const int32_t* origin_height = origin_height_blob->dptr<int32_t>();
  const int32_t* origin_width = BnInOp2Blob("origin_width")->dptr<int32_t>();
  T* scale = BnInOp2Blob("scale")->mut_dptr<T>();
  int32_t* resized_image_size = BnInOp2Blob("resized_image_size")->mut_dptr<int32_t>();
  int32_t* aligned_image_size = BnInOp2Blob("aligned_image_size")->mut_dptr<int32_t>();
  const auto& conf = this->op_conf().calculate_image_size_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();
  const int32_t dynamic_shape_align = conf.dynamic_shape_align();

  const int32_t num_images = origin_height_blob->shape().elem_cnt();
  FOR_RANGE(int32_t, i, 0, num_images) {
    const int32_t im_size_min = std::min(origin_height[i], origin_width[i]);
    const int32_t im_size_max = std::max(origin_height[i], origin_width[i]);
    scale[i] = static_cast<T>(target_size) / static_cast<T>(im_size_min);
    if (std::round(scale[i] * im_size_max) > max_size) {
      scale[i] = static_cast<T>(max_size) / static_cast<T>(im_size_max);
    }
    resized_image_size[2 * i] = static_cast<int32_t>(origin_height[i] * scale[i]);
    resized_image_size[2 * i + 1] = static_cast<int32_t>(origin_width[i] * scale[i]);

    // recalculate scales in y-direction and x-direction
    scale[2 * i] =
        static_cast<float>(resized_image_size[2 * i]) / static_cast<float>(origin_height[i]);
    scale[2 * i + 1] =
        static_cast<float>(resized_image_size[2 * i + 1]) / static_cast<float>(origin_width[i]);
  }

  // calculate dynamic shape
  int32_t batch_height = -1;
  int32_t batch_width = -1;
  FOR_RANGE(int32_t, i, 0, num_images) {
    batch_height = std::max(batch_height, resized_image_size[2 * i]);
    batch_width = std::max(batch_width, resized_image_size[2 * i + 1]);
  }
  aligned_image_size[0] = RoundUp(batch_height, dynamic_shape_align);
  aligned_image_size[1] = RoundUp(batch_width, dynamic_shape_align);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCalculateImageSizeConf, CalculateImageSizeKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
