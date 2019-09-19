#ifndef ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SliceBoxingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingKernel);
  SliceBoxingKernel() = default;
  ~SliceBoxingKernel() override = default;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const = 0;
  MemoryCopier* memory_copier() const;
  const std::vector<std::shared_ptr<TensorSliceCopier>>& tensor_slice_copier_vec() const;

 private:
  void VirtualKernelInit(const ParallelContext*);

  std::vector<std::shared_ptr<TensorSliceCopier>> tensor_slice_copier_vec_;
  std::unique_ptr<MemoryCopier> memory_copier_;
};

template<DeviceType device_type, typename T>
class SliceBoxingCopyKernel final : public SliceBoxingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingCopyKernel);
  SliceBoxingCopyKernel() = default;
  ~SliceBoxingCopyKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class SliceBoxingAddKernel final : public SliceBoxingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingAddKernel);
  SliceBoxingAddKernel() = default;
  ~SliceBoxingAddKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_H_
