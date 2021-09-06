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
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace {

template<typename T, typename K>
__global__ void ComputeNllOut(const int64_t num_instances, const K num_classes, const T* input,
                              const K* target, const K ignore_index, T* out) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(target[i] >= 0);
    assert(target[i] < num_classes);
    K label = target[i];
    if (label == ignore_index) {
      out[i] = 0;
      continue;
    }
    out[i] = -input[i * num_classes + label];
  }
}
template<typename K>
__global__ void ComputeNllOutHalf(const int64_t num_instances, const K num_classes,
                                  const half* input, const K* target, const K ignore_index,
                                  half* out) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(target[i] >= 0);
    assert(target[i] < num_classes);
    K label = target[i];
    if (label == ignore_index) {
      out[i] = 0;
      continue;
    }
    out[i] = __float2half(-__half2float(input[i * num_classes + label]));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T, typename K>
__global__ void ComputeNllGradOut(const int64_t num_instances, const K num_classes, const T* dy,
                                  const K* target, const K ignore_index, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(target[i] >= 0);
    assert(target[i] < num_classes);
    K label = target[i];
    if (label == ignore_index) { continue; }
    dx[i * num_classes + label] = -dy[i];
  }
}

template<typename K>
__global__ void ComputeNllGradOutHalf(const int64_t num_instances, const K num_classes,
                                      const half* dy, const K* target, const K ignore_index,
                                      half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    assert(target[i] >= 0);
    assert(target[i] < num_classes);
    K label = target[i];
    if (label == ignore_index) { continue; }
    dx[i * num_classes + label] = __float2half(-__half2float(dy[i]));
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
    // auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);
    const int64_t num_instances = target_blob->shape().elem_cnt();
    CHECK_EQ(input_blob->shape().elem_cnt() % num_instances, 0);
    const K num_classes = static_cast<K>(input_blob->shape().elem_cnt() / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const T* input = input_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    T* out = out_blob->mut_dptr<T>();

    ComputeNllOut<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                    ctx->device_ctx()->cuda_stream()>>>(num_instances, num_classes, input, target,
                                                        ignore_index, out);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<typename K>
class NllKernel<float16, K> final : public user_op::OpKernel {
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
    const float16* input = input_blob->dptr<float16>();
    const K* target = target_blob->dptr<K>();
    float16* out = out_blob->mut_dptr<float16>();

    ComputeNllOutHalf<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(
        num_instances, num_classes, reinterpret_cast<const half*>(input), target, ignore_index,
        reinterpret_cast<half*>(out));
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
    // auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);
    const int64_t num_instances = target_blob->shape().elem_cnt();
    const int64_t input_elem_cnt = input_blob->shape().elem_cnt();
    CHECK_EQ(input_elem_cnt % num_instances, 0);
    const K num_classes = static_cast<K>(input_elem_cnt / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const T* dy = dy_blob->dptr<T>();
    const K* target = target_blob->dptr<K>();
    T* dx = dx_blob->mut_dptr<T>();
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx, 0,
                             GetCudaAlignedSize(input_elem_cnt * sizeof(T)));

    ComputeNllGradOut<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(num_instances, num_classes, dy, target,
                                                            ignore_index, dx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename K>
class NllGradKernel<float16, K> final : public user_op::OpKernel {
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
    // auto* total_weight_blob = ctx->Tensor4ArgNameAndIndex("total_weight", 0);
    const int64_t num_instances = target_blob->shape().elem_cnt();
    const int64_t input_elem_cnt = input_blob->shape().elem_cnt();
    CHECK_EQ(input_elem_cnt % num_instances, 0);
    const K num_classes = static_cast<K>(input_elem_cnt / num_instances);
    const K ignore_index = static_cast<K>(ctx->Attr<int64_t>("ignore_index"));
    const float16* dy = dy_blob->dptr<float16>();
    const K* target = target_blob->dptr<K>();
    float16* dx = dx_blob->mut_dptr<float16>();
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx, 0,
                             GetCudaAlignedSize(input_elem_cnt * sizeof(float16)));

    ComputeNllGradOutHalf<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(
        num_instances, num_classes, reinterpret_cast<const half*>(dy), target, ignore_index,
        reinterpret_cast<half*>(dx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
#define REGISTER_NLL_KERNEL(dtype_pair, ltype_pair)                                           \
  REGISTER_USER_KERNEL("nll")                                                                 \
      .SetCreateFn<NllKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                          \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair)) \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

#define REGISTER_NLL_GRAD_KERNEL(dtype_pair, ltype_pair)                                        \
  REGISTER_USER_KERNEL("nll_grad")                                                              \
      .SetCreateFn<NllGradKernel<OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(ltype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                            \
                       & (user_op::HobDataType("target", 0) == OF_PP_PAIR_SECOND(ltype_pair))   \
                       & (user_op::HobDataType("dy", 0) == OF_PP_PAIR_SECOND(dtype_pair))       \
                       & (user_op::HobDataType("dx", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_KERNEL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_NLL_GRAD_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
