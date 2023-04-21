/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// #include "oneflow/user/kernels/normal_with_tensor_kernel_util.h"

// namespace oneflow {

// // Functor for adding two tensors
// template<typename T>
// struct  GpuNormalTensorFloatFunctor {
//   OF_DEVICE_FUNC T operator()(T output_val, T mean_val) const {
//     // Add the two input values and return the result
//     return output_val + mean_val;
//   }
// };

// template<DeviceType device_type, typename T>
// class GpuNormalTensorFloatKernel final : public user_op::OpKernel {
//  public:
//   GpuNormalTensorFloatKernel() = default;
//   ~GpuNormalTensorFloatKernel() = default;

//   std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
//       user_op::KernelInitContext* ctx) const override {
//     const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
//     // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
//     generator->set_current_seed(
//         CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
//     return std::make_shared<DistributionKernelState>(generator);
//   }

//  private:
//   void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
//                const user_op::OpKernelCache*) const override {
//     const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
//     const double std = ctx->Attr<double>("std");

//     user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
//     const int32_t elem_cnt = mean->shape_view().elem_cnt();
//     T* out_dptr = out->mut_dptr<T>();
//     auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
//     CHECK_NOTNULL(distribution_state);
//     const auto& generator = distribution_state->generator();
//     CHECK_NOTNULL(generator);
//     NormalDistribution<device_type, T> distribution(static_cast<T>(0), static_cast<T>(std));
//     distribution(ctx->stream(), elem_cnt, out_dptr, generator);

//     // Use CUDA Elementwise Template.
//     // OF_CUDA_CHECK((cuda::elementwise::Binary(GpuNormalTensorFloatFunctor<T>(), elem_cnt,
//     out_dptr,
//     //                                     out_dptr , mean->dptr<T>(),
//     ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

//   }
//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
// };

// #define REGISTER_CUDA_NORMAL_TENSOR_FLOAT_KERNEL(device,dtype)                              \
//   REGISTER_USER_KERNEL("normal_tensor_float")                                               \
//       .SetCreateFn<GpuNormalTensorFloatKernel<device,dtype>>()                             \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                                 \
//                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

// }  // namespace oneflow
