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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace user_op {

namespace {

constexpr int32_t kBlockSize = 1024; 

template<typename T>
__global__ void FusedBinaryCrossEntropyWithLogitsReduceMeanKernel(const T* input, const T* target, T* out, const int32_t elem_cnt){
    T zero = static_cast<T>(0.0); 
    T one = static_cast<T>(1.0); 
    using BlockReduce = cub::BlockReduce<T, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T reduce_sum = 0.0; 
    CUDA_1D_KERNEL_LOOP(i, elem_cnt){
        const T input_val = input[i]; 
        const T target_val = target[i]; 
        const T max_val = -input_val < zero ? zero : -input_val;
        const T result = (one - target_val) * input_val + max_val + (log(exp(-max_val) + exp(-input_val - max_val)));
        const T block_reduce_sum = BlockReduce(temp_storage).Sum(result);
        if (threadIdx.x == 0) { reduce_sum += block_reduce_sum; }
    }
    if (threadIdx.x == 0) { out[0] = reduce_sum / elem_cnt; }
}

template<>
__global__ void FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<half>(const half* input, const half* target, half* out, const int32_t elem_cnt){
    float zero = static_cast<float>(0.0); 
    float one = static_cast<float>(1.0); 
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float reduce_sum = 0.0; 
    CUDA_1D_KERNEL_LOOP(i, elem_cnt){
        const float input_val = __half2float(input[i]); 
        const float target_val = __half2float(target[i]); 
        const float max_val = -input_val < zero ? zero : -input_val;
        const float result = (one - target_val) * input_val + max_val + (log(exp(-max_val) + exp(-input_val - max_val)));
        const float block_reduce_sum = BlockReduce(temp_storage).Sum(result);
        if (threadIdx.x == 0) { reduce_sum += block_reduce_sum; }
    }
    if (threadIdx.x == 0) { out[0] = __float2half(reduce_sum / elem_cnt); }
}

// template<typename T>
// __device__ __forceinline__ T CalSigmoid(const T x) {
//   const T half_of_one = static_cast<T>(0.5);
//   return half_of_one * tanh(half_of_one * x) + half_of_one;
// }

// template<>
// __device__ __forceinline__ float CalSigmoid(const float x) {
//   const float half_of_one = static_cast<float>(0.5);
//   return half_of_one * tanhf(half_of_one * x) + half_of_one;
// }

// template<>
// __device__ __forceinline__ half CalSigmoid(const half x) {
//   return __float2half(CalSigmoid(__half2float(x)));
// }

// template<typename T, WeightType WEIGHT_TYPE>
// struct BinaryCrossEntropyWithLogitsGradFunctor;

// template<typename T>
// struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kNone> {
//   __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val) const {
//     return (CalSigmoid(input_val) - target_val) * dy_val;
//   }
// };

// template<typename T>
// struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kPosWeight> {
//   T one_;
//   BinaryCrossEntropyWithLogitsGradFunctor() : one_(GetOneVal<T>()) {}
//   __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val) const {
//     return dy_val * ((weight_val + one_ - target_val) * CalSigmoid(input_val) - weight_val);
//   }
// };

// template<typename T>
// struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kWeight> {
//   BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kNone> f;
//   __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val) const {
//     return f(input_val, target_val, dy_val) * weight_val;
//   }
// };

// template<typename T>
// struct BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kBoth> {
//   BinaryCrossEntropyWithLogitsGradFunctor<T, WeightType::kPosWeight> f;
//   __device__ __forceinline__ T operator()(T input_val, T target_val, T dy_val, T weight_val,
//                                           T pos_weight_val) const {
//     return f(input_val, target_val, dy_val, pos_weight_val) * weight_val;
//   }
// };

template<typename T>
class BinaryCrossEntropyWithLogitsMeanKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsMeanKernel() = default;
  ~BinaryCrossEntropyWithLogitsMeanKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t elem_cnt = input_blob->shape().elem_cnt();

    const T* input = input_blob->dptr<T>();
    const T* target = target_blob->dptr<T>();
    T* out = out_blob->mut_dptr<T>();

    FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<<<1, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        input_blob->dptr<T>(), target_blob->dptr<T>(), out_blob->mut_dptr<T>(), input_blob->shape().elem_cnt()
    ); 
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_reduce_mean")                                 \
      .SetCreateFn<BinaryCrossEntropyWithLogitsMeanKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
                       && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

// #define REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(dtype)                                   \
//   REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                            \
//       .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<dtype>>()                        \
//       .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
//                        && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)  \
//                        && (user_op::HobDataType("target", 0) == GetDataType<dtype>::value) \
//                        && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value)     \
//                        && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))    \
//       .SetInferTmpSizeFn(GenBwInferTmpSizeFn<dtype>());

// REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(double)

// REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(half)
// REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(float)
// REGISTER_BINARY_CROSS_ENTROPY_GRAD_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
