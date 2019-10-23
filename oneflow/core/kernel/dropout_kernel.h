#ifndef ONEFLOW_CORE_KERNEL_DROPOUT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DROPOUT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class DropoutKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutKernel);
  DropoutKernel() = default;
  ~DropoutKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class DropoutGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutGradKernel);
  DropoutGradKernel() = default;
  ~DropoutGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
class RandomMaskLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomMaskLikeKernel);
  RandomMaskLikeKernel() = default;
  ~RandomMaskLikeKernel() = default;

 private:
  void VirtualKernelInit(DeviceCtx*) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<RandomGenerator<device_type>> random_generator_;
};

template<DeviceType device_type, typename T, typename U = void>
struct DropoutKernelUtil final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x,
                           const int8_t* mask, T* y);
};

template<DeviceType device_type>
struct RandomMaskLikeKernelUtil final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
                      int8_t* mask);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DROPOUT_KERNEL_H_
