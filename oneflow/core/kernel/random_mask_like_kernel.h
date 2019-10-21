#ifndef ONEFLOW_CORE_KERNEL_RANDOM_MASK_LIKE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_MASK_LIKE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

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

template<DeviceType device_type>
struct RandomMaskLikeKernelUtil final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp, int8_t* mask);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_MASK_LIKE_KERNEL_H_
