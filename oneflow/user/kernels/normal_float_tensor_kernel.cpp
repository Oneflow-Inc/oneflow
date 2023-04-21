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
// /*
// Copyright 2020 The OneFlow Authors. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */
// #include "oneflow/core/device/cuda_util.h"
// #include "oneflow/core/framework/framework.h"
// #include "oneflow/core/cuda/elementwise.cuh"
// #include "oneflow/user/kernels/normal_with_tensor_kernel_util.h"

// namespace oneflow {
// namespace  {
// template<typename T>
// struct GpuNormalFloatTensorFunctor {
//   T std_;
//   explicit GpuNormalFloatTensorFunctor(T std) : std_(std) {}
//   OF_DEVICE_FUNCTION T operator()(T output_val, T mean_val) const {
//     // Add the two input values and return the result
//     return mean_val + output_val * std_;
//   }
// };

// template<DeviceType device_type, typename T>
// class CudaNormalFloatTensorKernel final : public user_op::OpKernel {
//  public:
//   CudaNormalFloatTensorKernel() = default;
//   ~CudaNormalFloatTensorKernel() = default;
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
//     const double mean = ctx->Attr<double>("mean");
//     const user_op::Tensor* std = ctx->Tensor4ArgNameAndIndex("std", 0);
//     user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

//     normal_out_impl<device_type, T>(ctx, state, out, mean, std);

//     T* out_dptr = out->mut_dptr<T>();
//     const int64_t elem_cnt = std->shape_view().elem_cnt();
//     cudaMemcpy(out_copy->mut_dptr(), out->dptr(), out->shape().elem_cnt() * sizeof(T),
//     cudaMemcpyDeviceToDevice); GpuNormalFloatTensorFunctor<T> func(static_cast<T>(mean));
//     OF_CUDA_CHECK((cuda::elementwise::Unary(func, elem_cnt, out_dptr,
//                                           std->dptr<T>(),
//                                         ctx->stream()->As<ep::CudaStream>()->cuda_stream())));

//   }
//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
// };

// #define REGISTER_CUDA_NORMAL_FLOAT_TENSOR_KERNEL(device, dtype)  \
//   REGISTER_USER_KERNEL("normal_float_tensor")                     \
//       .SetCreateFn<CudaNormalFloatTensorKernel<device, dtype>>() \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)       \
//                        && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

// REGISTER_CUDA_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCUDA, half)
// REGISTER_CUDA_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCUDA, float)
// REGISTER_CUDA_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCUDA, double)
// } // namespace

<<<<<<< HEAD
// }  // namespace oneflow
// /*
// +---------------------+
// | Retrieve input data |
// +---------------------+
//            |
//            v
// +------------------------+
// | Compute normal dist.    |
// +------------------------+
//            |
//            v
// +------------------------+
// | Compute output          |
// | using CUDA Elementwise  |
// | Template                |
// +------------------------+
// */
=======
}  // namespace oneflow
>>>>>>> 30c8c177857ed8f2855cb7954a1a8426636a7a0d
