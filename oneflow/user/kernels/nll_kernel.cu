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
#include <cub/cub.cuh>
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

#define RETURN_VOID_IF_NOT_HALF typename std::enable_if_t<!std::is_same<T, half>::value, void>
#define RETURN_VOID_IF_HALF typename std::enable_if_t<std::is_same<T, half>::value, void>

template<typename T, typename K>
__global__ RETURN_VOID_IF_NOT_HALF ComputeNllOutNone(const int64_t num_instances,
                                                     const K num_classes, const K ignore_index,
                                                     const T* input, const K* target, T* out,
                                                     const T* weight, T* total_weight) {
  const T zero_val = GetZeroVal<T>();
  const T one_val = GetOneVal<T>();
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = target[i];
    if (label == ignore_index) {
      out[i] = zero_val;
      continue;
    }
    assert(label >= 0);
    assert(label < num_classes);
    const T cur_weight = weight == nullptr ? one_val : weight[label];
    cuda::atomic::Add(total_weight, cur_weight);
    out[i] = -input[i * num_classes + label] * cur_weight;
  }
}

template<typename T, typename K>
__global__ RETURN_VOID_IF_HALF ComputeNllOutNone(const int64_t num_instances, const K num_classes,
                                                 const K ignore_index, const T* input,
                                                 const K* target, T* out, const T* weight,
                                                 T* total_weight) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const T zero_val = __float2half(0.0);
  const T one_val = __float2half(1.0);
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = target[i];
    if (label == ignore_index) {
      out[i] = zero_val;
      continue;
    }
    assert(label >= 0);
    assert(label < num_classes);
    const half cur_weight = weight == nullptr ? one_val : weight[label];
    cuda::atomic::Add(total_weight, cur_weight);
    out[i] = __float2half(-__half2float(input[i * num_classes + label] * cur_weight));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ RETURN_VOID_IF_NOT_HALF ComputeNllOutReduce(const int64_t num_instances,
                                                       const K num_classes, const K ignore_index,
                                                       const T* input, const K* target, T* out,
                                                       const T* weight, T* total_weight,
                                                       bool is_reduce_mean) {
  const T zero_val = GetZeroVal<T>();
  const T one_val = GetOneVal<T>();
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  T weight_thread_sum = zero_val;
  T out_thread_sum = zero_val;
  for (int i = threadIdx.x; i < num_instances; i += kCudaThreadsNumPerBlock) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    assert(label >= 0);
    assert(label < num_classes);
    const T cur_weight = weight == nullptr ? one_val : weight[label];
    weight_thread_sum += cur_weight;
    out_thread_sum -= input[i * num_classes + label] * cur_weight;
  }
  __syncthreads();
  T weight_block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(weight_thread_sum, cub::Sum());
  T out_block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(out_thread_sum, cub::Sum());
  if (threadIdx.x == 0) {
    *out = out_block_sum;
    *total_weight = weight_block_sum;
    if (is_reduce_mean) { *out /= *total_weight; }
  }
}

template<typename T, typename K>
__global__ RETURN_VOID_IF_HALF ComputeNllOutReduce(const int64_t num_instances, const K num_classes,
                                                   const K ignore_index, const T* input,
                                                   const K* target, T* out, const T* weight,
                                                   T* total_weight, bool is_reduce_mean) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  const T zero_val = __float2half(0.0);
  const T one_val = __float2half(1.0);
  typedef cub::BlockReduce<half, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  half weight_thread_sum = zero_val;
  half out_thread_sum = zero_val;
  for (int i = threadIdx.x; i < num_instances; i += kCudaThreadsNumPerBlock) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    assert(label >= 0);
    assert(label < num_classes);
    const half cur_weight = weight == nullptr ? one_val : weight[label];
    weight_thread_sum = __hadd(weight_thread_sum, cur_weight);
    out_thread_sum = __hsub(out_thread_sum, __hmul(input[i * num_classes + label], cur_weight));
  }
  __syncthreads();
  half weight_block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(weight_thread_sum, cub::Sum());
  half out_block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(out_thread_sum, cub::Sum());
  if (threadIdx.x == 0) {
    *out = out_block_sum;
    *total_weight = weight_block_sum;
    if (is_reduce_mean) { *out = __hdiv(*out, *total_weight); }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ RETURN_VOID_IF_NOT_HALF ComputeNllGradOut(const int64_t num_instances,
                                                     const K num_classes, const K ignore_index,
                                                     const K* target, const T* dy, T* dx,
                                                     const T* weight, const T* total_weight,
                                                     const ReductionType reduction_type) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    assert(label >= 0);
    assert(label < num_classes);
    const T cur_weight = weight == nullptr ? -GetOneVal<T>() : -weight[label];
    dx[i * num_classes + label] =
        (reduction_type == ReductionType::kNone ? dy[i] : (*dy)) * cur_weight;
    if (reduction_type == ReductionType::kMean) { dx[i * num_classes + label] /= *total_weight; }
  }
}

template<typename T, typename K>
__global__ RETURN_VOID_IF_HALF ComputeNllGradOut(const int64_t num_instances, const K num_classes,
                                                 const K ignore_index, const K* target, const T* dy,
                                                 T* dx, const T* weight, const T* total_weight,
                                                 const ReductionType reduction_type) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    K label = target[i];
    if (label == ignore_index) { continue; }
    assert(label >= 0);
    assert(label < num_classes);
    const half cur_weight = weight == nullptr ? __float2half(-1.0) : __hneg(weight[label]);
    dx[i * num_classes + label] =
        __hmul(reduction_type == ReductionType::kNone ? dy[i] : (*dy), cur_weight);
    if (reduction_type == ReductionType::kMean) {
      dx[i * num_classes + label] = __hdiv(dx[i * num_classes + label], *total_weight);
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
class NllKernel final : public user_op::OpKernel {
 public:
  NllKernel() = default;
  ~NllKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);

    const int64_t num_instances = target_blob->shape().elem_cnt();
    CHECK_EQ(input_blob->shape().elem_cnt() % num_instances, 0);
    const K num_classes = static_cast<K>(input_blob->shape().elem_cnt() / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const T* input = input_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    T* out = out_blob->mut_dptr<T>();
    T* total_weight = total_weight_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;
    Memset<DeviceType::kGPU>(ctx->device_ctx(), total_weight, 0, sizeof(T));

    if (reduction == ReductionType::kNone) {
      ComputeNllOutNone<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                          ctx->device_ctx()->cuda_stream()>>>(
          num_instances, num_classes, ignore_index, input, target, out, weight, total_weight);
    } else {
      ComputeNllOutReduce<<<1, kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
          num_instances, num_classes, ignore_index, input, target, out, weight, total_weight,
          reduction == ReductionType::kMean);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename K>
class NllGradKernel final : public user_op::OpKernel {
 public:
  NllGradKernel() = default;
  ~NllGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);

    const int64_t num_instances = target_blob->shape().elem_cnt();
    const int64_t input_elem_cnt = input_blob->shape().elem_cnt();
    CHECK_EQ(input_elem_cnt % num_instances, 0);
    const K num_classes = static_cast<K>(input_elem_cnt / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>("reduction"));

    const T* dy = dy_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    const T* total_weight = total_weight_blob->dptr<T>();
    T* dx = dx_blob->mut_dptr<T>();
    const T* weight =
        ctx->has_input("weight", 0) ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<T>() : nullptr;

    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx, 0,
                             GetCudaAlignedSize(input_elem_cnt * sizeof(T)));

    ComputeNllGradOut<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(
        num_instances, num_classes, ignore_index, target, dy, dx, weight, total_weight, reduction);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
#define REGISTER_NLL_KERNEL(dtype_pair, ltype_pair)                                           \
  REGISTER_USER_KERNEL("nll")                                                                 \
      .SetCreateFn<NllKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>()   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                         \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair)) \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

#define REGISTER_NLL_GRAD_KERNEL(dtype_pair, ltype_pair)                                        \
  REGISTER_USER_KERNEL("nll_grad")                                                              \
      .SetCreateFn<NllGradKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                           \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair))   \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dtype_pair))       \
                       & (user_op::HobDataType("dx", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_KERNEL, FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_GRAD_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
