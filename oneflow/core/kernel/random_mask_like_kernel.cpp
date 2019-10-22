#include "oneflow/core/kernel/random_mask_like_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<DeviceType device_type>
void RandomMaskLikeKernel<device_type>::VirtualKernelInit(DeviceCtx* device_ctx) {
  const auto& random_mask_like_conf = this->op_conf().random_mask_like_conf();
  int64_t seed = GetCurTime();
  if (random_mask_like_conf.has_seed()) { seed = random_mask_like_conf.seed(); }
  random_generator_.reset(new RandomGenerator<device_type>(seed, device_ctx));
}

template<DeviceType device_type>
void RandomMaskLikeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->job_desc().IsTrain()) {
    int64_t elem_cnt = BnInOp2Blob("out")->shape().elem_cnt();
    float* random_tmp = BnInOp2Blob("random_tmp")->mut_dptr<float>();
    int8_t* mask = BnInOp2Blob("out")->mut_dptr<int8_t>();
    random_generator_->Uniform(elem_cnt, random_tmp);
    RandomMaskLikeKernelUtil<device_type>::GenMask(
        ctx.device_ctx, elem_cnt, this->op_conf().random_mask_like_conf().rate(), random_tmp, mask);
  } else {
    // do nothing
  }
}

template<>
struct RandomMaskLikeKernelUtil<DeviceType::kCPU> final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
                      int8_t* mask) {
    for (int64_t i = 0; i < n; ++i) { mask[i] = random_tmp[i] > threshold; }
  }
};

template struct RandomMaskLikeKernelUtil<DeviceType::kCPU>;

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kRandomMaskLikeConf, RandomMaskLikeKernel);

}  // namespace oneflow
