#ifndef ONEFLOW_CORE_KERNEL_TRANSPOSE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_TRANSPOSE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class TransposeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransposeKernel);
  TransposeKernel() = default;
  ~TransposeKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void Transpose(DeviceCtx* ctx, const Blob* in_blob, Blob* out_blob,
               const PbRf<int32_t>& permutation) {
  KernelUtil<device_type, T>::Transpose(
      ctx, in_blob->shape().NumAxes(), in_blob->shape(), out_blob->shape(),
      permutation, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
      out_blob->mut_dptr<T>());
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TRANSPOSE_KERNEL_H_
