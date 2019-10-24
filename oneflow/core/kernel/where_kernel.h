#ifndef ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class WhereKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereKernel);
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct WhereKernelUtil {
  static void Where(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* x_dptr,
                    const T* y_dptr, T* out_dptr);
  static void CmptXDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* x_diff_dptr);
  static void CmptYDiff(DeviceCtx* ctx, const int64_t n, const T* cond_dptr, const T* out_diff_dptr,
                        T* y_diff_dptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_WHERE_KERNEL_H_
