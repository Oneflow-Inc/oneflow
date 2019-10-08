#include "oneflow/core/kernel/level_map_kernel_util.h"

namespace oneflow {

template<typename T>
struct LevelMapUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t num_boxes, const T* in_ptr,
                      const int32_t canonical_level, const float canonical_scale,
                      const int32_t min_level, const int32_t max_level, const float epsilon,
                      int32_t* out_ptr) {
    const T TO_REMOVE = 1.0;
    FOR_RANGE(int64_t, i, 0, num_boxes) {
      const T scale = std::sqrt((in_ptr[i * 4 + 2] - in_ptr[i * 4] + TO_REMOVE)
                                * (in_ptr[i * 4 + 3] - in_ptr[i * 4 + 1] + TO_REMOVE));
      const int32_t target_level =
          std::floor(canonical_level + std::log2(scale / canonical_scale + epsilon));
      out_ptr[i] = std::min(std::max(target_level, min_level), max_level) - min_level;
    }
  }
};

template<DeviceType device_type, typename T>
class LevelMapKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LevelMapKernel);
  LevelMapKernel() = default;
  ~LevelMapKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");

    const auto in_dim_vec = static_cast<Shape>(in_blob->shape()).dim_vec();
    CHECK_EQ(in_dim_vec.back(), 4);
    int64_t num_boxes =
        std::accumulate(in_dim_vec.begin(), in_dim_vec.end() - 1, 1, std::multiplies<int64_t>());
    const auto conf = this->op_conf().level_map_conf();
    LevelMapUtil<device_type, T>::Forward(ctx.device_ctx, num_boxes, in_blob->dptr<T>(),
                                          conf.canonical_level(), conf.canonical_scale(),
                                          conf.min_level(), conf.max_level(), conf.epsilon(),
                                          out_blob->mut_dptr<int32_t>());
  }
};

#define REGISTER_LEVEL_MAP_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLevelMapConf, DeviceType::kCPU, dtype, \
                                        LevelMapKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLevelMapConf, DeviceType::kGPU, dtype, \
                                        LevelMapKernel<DeviceType::kGPU, dtype>)

REGISTER_LEVEL_MAP_KERNEL(float);
REGISTER_LEVEL_MAP_KERNEL(double);

}  // namespace oneflow
