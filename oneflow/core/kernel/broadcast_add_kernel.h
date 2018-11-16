#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_ADD_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct BroadcastAddKernelUtil final {
  static void BackwardInputDiff(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff_a,
                                const XpuVarNdarray<const T>& out_diff, XpuVarNdarray<T>&& bw_buf);
};

template<DeviceType device_type, typename T>
class BroadcastAddKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastAddKernel);
  BroadcastAddKernel() = default;
  ~BroadcastAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;

#define MAKE_FUNC_BACKWARD_INPUT_DIFF(func_name, NDIMS) \
  BroadcastAddKernelUtil<device_type, T, NDIMS>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, BackwardInputDiff, MAKE_FUNC_BACKWARD_INPUT_DIFF,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_FUNC_BACKWARD_INPUT_DIFF
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_ADD_KERNEL_H_
