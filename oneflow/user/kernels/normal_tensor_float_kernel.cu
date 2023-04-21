#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

// Functor for adding two tensors
template<typename T>
struct  CudaNormalTensorFloatFunctor {
  OF_DEVICE_FUNC T operator()(T output_val, T mean_val) const {
    // Add the two input values and return the result
    return output_val + mean_val;
  }
};

}  // namespace

template<DeviceType device_type, typename T>
class CudaNormalTensorFloatKernel final : public user_op::OpKernel {
 public:
  CudaNormalTensorFloatKernel() = default;
  ~CudaNormalTensorFloatKernel() = default;
 
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const double std = ctx->Attr<double>("std");
    
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = mean->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(0), static_cast<T>(std));
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);

    // Use CUDA Elementwise Template. 
    OF_CUDA_CHECK((cuda::elementwise::Binary(CudaNormalTensorFloatFunctor<T>(), elem_cnt, out_dptr,
                                        out_dptr , mean->dptr<T>(), ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_NORMAL_TENSOR_FLOAT_KERNEL(device,dtype)                              \
  REGISTER_USER_KERNEL("normal_tensor_float")                                       \
      .SetCreateFn<CpuNormalTensorFloatKernel<device,dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)               \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCUDA, half)
REGISTER_CUDA_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCUDA, float)
REGISTER_CUDA_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCUDA, double)

}  // namespace oneflow
