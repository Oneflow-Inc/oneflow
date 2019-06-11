#ifndef ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ClipByValueKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByValueKernel);
  ClipByValueKernel() = default;
  ~ClipByValueKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct ClipByValueUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* in_ptr, const T min_val,
                      const T max_val, int8_t* clip_mask_ptr, T* out_ptr);
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* out_diff_ptr,
                       const int8_t* clip_mask_ptr, T* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_KERNEL_OP_H_
