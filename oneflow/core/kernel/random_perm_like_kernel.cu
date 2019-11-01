#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void AtomicExchange(const int64_t n, T* in, const int64_t retry_limit) {
  CUDA_1D_KERNEL_LOOP(i, n) {}
}

}  // namespace

template<typename T>
class RandomPermLikeGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomPermLikeGPUKernel);
  RandomPermLikeGPUKernel() = default;
  ~RandomPermLikeGPUKernel() = default;

  void VirtualKernelInit(DeviceCtx* device_ctx) {
    const auto& dropout_conf = this->op_conf().dropout_conf();
    int64_t seed = -1;
    const RandomPermLikeOpConf& random_perm_like_conf = this->op_conf().random_perm_like_conf();
    if (random_perm_like_conf.has_random_seed()) {
      seed = random_perm_like_conf.random_seed();
    } else {
      seed = GetCurTime();
    }
    CHECK_NE(seed, -1);
    random_generator_.reset(new RandomGenerator<DeviceType::kGPU>(seed, device_ctx));
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("out");
    Blob* random_mask_blob = BnInOp2Blob("random_mask");
    const int64_t n = out_blob->shape().At(0);
    random_generator_->Uniform(n, random_mask_blob->mut_dptr<float>());
  }

 private:
  std::unique_ptr<RandomGenerator<DeviceType::kGPU>> random_generator_;
};

#define REGISTER_RANDOM_PERM_GPU_KERNEL(dtype)                                               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRandomPermLikeConf, DeviceType::kGPU, \
                                        dtype, RandomPermLikeGPUKernel<dtype>);

REGISTER_RANDOM_PERM_GPU_KERNEL(int32_t);

}  // namespace oneflow
