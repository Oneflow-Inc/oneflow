#include "oneflow/core/kernel/level_map_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LevelMapKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  CHECK_EQ(in_blob->shape().dim_vec().back(), 4);
  int64_t num_boxes =
      std::accumulate(in_blob->shape().dim_vec().begin(), in_blob->shape().dim_vec().end() - 1, 1,
                      std::multiplies<int64_t>());
  const auto conf = this->op_conf().level_map_conf();
  LevelMapUtil<device_type, T>::Forward(
      ctx.device_ctx, num_boxes, in_blob->dptr<T>(), conf.canonical_level(), conf.canonical_scale(),
      conf.min_level(), conf.max_level(), conf.epsilon(), out_blob->mut_dptr<int32_t>());
}

template<DeviceType device_type, typename T>
void LevelMapKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void LevelMapKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyInstanceShapeFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<typename T>
struct LevelMapUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t num_boxes, const T* in_ptr,
                      const int32_t canonical_level, const int32_t canonical_scale,
                      const int32_t min_level, const int32_t max_level, const float epsilon,
                      int32_t* out_ptr) {
    const T TO_REMOVE = 1.0;
    FOR_RANGE(int64_t, i, 0, num_boxes) {
      const T scale = std::sqrt((in_ptr[i * 4 + 2] - in_ptr[i * 4] + TO_REMOVE)
                                * (in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] + TO_REMOVE));
      const int32_t target_level =
          std::log2(canonical_level + log2(scale / canonical_scale + epsilon));
      out_ptr[i] = std::min(std::max(target_level, min_level), max_level) - canonical_level;
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLevelMapConf, LevelMapKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
