#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RandomPermKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomPermKernel);
  RandomPermKernel() = default;
  ~RandomPermKernel() = default;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const RandomPermOpConf& random_perm_conf = this->op_conf().random_perm_conf();
    Blob* out_blob = BnInOp2Blob("out");
    int64_t upper_bound = -1;
    if (random_perm_conf.has_like()) {
      const Blob* like_blob = BnInOp2Blob("like");
      upper_bound = like_blob->shape().At(0);
    } else if (random_perm_conf.has_upper_bound()) {
      upper_bound = random_perm_conf.upper_bound();
    }
    CHECK_GT(upper_bound, 0);
    std::vector<int32_t> result(upper_bound);
    std::iota(result.begin(), result.end(), 0);
    random_shuffle(result.begin(), result.end());
    MemoryCase host_mem_case;
    host_mem_case.mutable_host_mem();
    AutoMemcpy(ctx.device_ctx, out_blob->mut_dptr(), result.data(), out_blob->ByteSizeOfBlobBody(),
               out_blob->mem_case(), host_mem_case);
  }
};

#define REGISTER_RANDOM_PERM_KERNEL(device_type)                                             \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRandomPermConf, device_type, int32_t, \
                                        RandomPermKernel<device_type, int32_t>);             \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRandomPermConf, device_type, int32_t, \
                                        RandomPermKernel<device_type, int32_t>);

REGISTER_RANDOM_PERM_KERNEL(DeviceType::kCPU);
REGISTER_RANDOM_PERM_KERNEL(DeviceType::kGPU);

}  // namespace oneflow
