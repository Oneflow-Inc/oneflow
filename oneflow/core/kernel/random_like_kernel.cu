#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

template<typename T>
class RandomLikeGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomLikeGPUKernel);
  RandomLikeGPUKernel() = default;
  ~RandomLikeGPUKernel() = default;

  void VirtualKernelInit(DeviceCtx* device_ctx) {
    int64_t seed = GetCurTime();
    const RandomLikeOpConf& random_like_conf = this->op_conf().random_like_conf();
    if (random_like_conf.has_random_seed()) { seed = random_like_conf.random_seed(); }
    random_generator_.reset(new RandomGenerator<DeviceType::kGPU>(seed, device_ctx));
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("out");
    random_generator_->Uniform(out_blob->shape().elem_cnt(), out_blob->mut_dptr<float>());
  }

 private:
  std::unique_ptr<RandomGenerator<DeviceType::kGPU>> random_generator_;
};

#define REGISTER_RANDOM_GPU_KERNEL(dtype)                                                       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRandomLikeConf, DeviceType::kGPU, dtype, \
                                        RandomLikeGPUKernel<dtype>);

REGISTER_RANDOM_GPU_KERNEL(float);

}  // namespace oneflow
