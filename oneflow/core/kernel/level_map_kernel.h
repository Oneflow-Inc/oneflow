#ifndef ONEFLOW_CORE_OPERATOR_LEVEL_MAP_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_LEVEL_MAP_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LevelMapKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LevelMapKernel);
  LevelMapKernel() = default;
  ~LevelMapKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx&,
                            std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct LevelMapUtil {
  static void Forward(DeviceCtx* ctx, const int64_t num_boxes, const T* in_ptr,
                      const int32_t canonical_level, const int32_t canonical_scale,
                      const int32_t min_level, const int32_t max_level, const float epsilon,
                      int32_t* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LEVEL_MAP_KERNEL_OP_H_
