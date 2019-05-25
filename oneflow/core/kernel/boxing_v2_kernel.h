#ifndef ONEFLOW_CORE_KERNEL_BOXING_V2_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_V2_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxingV2Kernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2Kernel);
  BoxingV2Kernel() = default;
  ~BoxingV2Kernel() override = default;

 protected:
  virtual const BoxingV2Conf& GetCustomizedBoxingConf() const = 0;
  MemoryCopier* memory_copier() const;
  const std::vector<std::shared_ptr<TensorSliceCopier>>& tensor_slice_copier_vec() const;

 private:
  void VirtualKernelInit(const ParallelContext*);

  std::vector<std::shared_ptr<TensorSliceCopier>> tensor_slice_copier_vec_;
  std::unique_ptr<MemoryCopier> memory_copier_;
};

template<DeviceType device_type, typename T>
class BoxingV2CopyKernel final : public BoxingV2Kernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2CopyKernel);
  BoxingV2CopyKernel() = default;
  ~BoxingV2CopyKernel() override = default;

 private:
  virtual const BoxingV2Conf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class BoxingV2AddKernel final : public BoxingV2Kernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2AddKernel);
  BoxingV2AddKernel() = default;
  ~BoxingV2AddKernel() override = default;

 private:
  virtual const BoxingV2Conf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_V2_KERNEL_H_
