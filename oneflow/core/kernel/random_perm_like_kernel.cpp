#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RandomPermLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomPermLikeKernel);
  RandomPermLikeKernel() = default;
  ~RandomPermLikeKernel() = default;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("out");
    const Blob* like_blob = BnInOp2Blob("like");
    std::vector<T> result(like_blob->shape().At(0));
    std::random_device rd;
    std::mt19937 gen_(rd());
    std::iota(result.begin(), result.end(), 0);
    std::shuffle(result.begin(), result.end(), gen_);
    MemoryCase host_mem_case;
    host_mem_case.mutable_host_mem();
    AutoMemcpy(ctx.device_ctx, out_blob->mut_dptr(), result.data(), out_blob->ByteSizeOfBlobBody(),
               out_blob->mem_case(), host_mem_case);
  }
};

#define REGISTER_RANDOM_PERM_KERNEL(device_type)                                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kRandomPermLikeConf, device_type, int32_t, \
                                        RandomPermLikeKernel<device_type, int32_t>);

REGISTER_RANDOM_PERM_KERNEL(DeviceType::kCPU);

}  // namespace oneflow
