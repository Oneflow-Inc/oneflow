#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct BroadcastSubKernelUtil final {
  static void BackwardInputDiffA(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& tmp_storage);
  static void BackwardInputDiffB(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& tmp_storage);
};

template<DeviceType device_type, typename T>
class BroadcastSubKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSubKernel);
  BroadcastSubKernel() = default;
  ~BroadcastSubKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
#define MAKE_FUNC_BACKWARD_INPUT_DIFF_A(func_name, NDIMS) \
  BroadcastSubKernelUtil<device_type, T, NDIMS>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, BackwardInputDiffA, MAKE_FUNC_BACKWARD_INPUT_DIFF_A,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_FUNC_BACKWARD_INPUT_DIFF_A
#define MAKE_FUNC_BACKWARD_INPUT_DIFF_B(func_name, NDIMS) \
  BroadcastSubKernelUtil<device_type, T, NDIMS>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, BackwardInputDiffB, MAKE_FUNC_BACKWARD_INPUT_DIFF_B,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef MAKE_FUNC_BACKWARD_INPUT_DIFF_B
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
