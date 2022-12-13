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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/multi_reduce_kernel_util.h"
#include "oneflow/user/kernels/fused_clip_grad_util.h"
#include <cmath>

namespace oneflow {

namespace {

size_t InferFusedClipGradTempStorageSize(user_op::InferContext* ctx) {
  auto input_size = ctx->input_size("model_diff");
  std::cout << "input_size:" << input_size << std::endl;
  if (input_size == 0) { return 0; }
  int64_t max_elem_cnt = 0;
  int64_t pack_size = 0;
  int32_t num_blocks = 0;
  for (size_t i = 0; i < input_size; ++i) {
    int64_t elem_cnt = ctx->InputShape("model_diff", i).elem_cnt();
    std::cout << "shape:" << elem_cnt << std::endl;
    max_elem_cnt = std::max(max_elem_cnt, elem_cnt);
    pack_size++;
    if (pack_size == kMultiReduceScaleMulPackSize || i == input_size - 1) {
      CHECK_LT(max_elem_cnt, std::numeric_limits<int32_t>::max());
      num_blocks += BlocksNum4ThreadsNum(static_cast<int32_t>(max_elem_cnt));
      max_elem_cnt = 0;
      pack_size = 0;
    }
  }
  CHECK_LT(num_blocks, kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock * kCudaThreadsNumPerBlock)
      << "Too much blocks needed for computing " << ctx->op_name() << ", should be less than "
      << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock << "*" << kCudaThreadsNumPerBlock
      << ", but got " << num_blocks;
  size_t elem_size = GetSizeOfDataType(ctx->InputDType("model_diff", 0));
  return GetCudaAlignedSize(num_blocks * elem_size * 2);
}

template<DeviceType device_type, typename T>
class FusedClipGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedClipGradKernel() = default;
  ~FusedClipGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t input_size = ctx->input_size("model_diff");
    const float max_norm = ctx->Attr<float>("max_norm");
    const float norm_type = ctx->Attr<float>("norm_type");

    std::vector<MultiReduceParam<T>> params;
    params.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("model_diff", i);
      params[i].size = x->shape_view().elem_cnt();
      params[i].data = x->dptr<T>();
    }

    T* temp = (ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0))->mut_dptr<T>();
    T *total_norm = nullptr;
    OF_CUDA_CHECK(cudaMalloc(&total_norm, sizeof(T)));

    bool not_special = false;
    if (norm_type == 0) {
      PowByZero<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_add{};
      reduce_add(ctx->stream(), func, params, GetZeroVal<T>(), total_norm, temp);
    } else if (norm_type == INFINITY) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryMax<T>> reduce_max{};
      reduce_max(ctx->stream(), func, params, GetZeroVal<T>(), total_norm, temp);
    } else if (norm_type == -INFINITY) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryMin<T>> reduce_min{};
      reduce_min(ctx->stream(), func, params, std::numeric_limits<T>::max(), total_norm, temp);
    } else if (norm_type == 1) {
      Abs<T> func{};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), total_norm, temp);
    } else {
      not_special = true;
      AbsPow<T> func{norm_type};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), total_norm, temp);
    }

    T *h_total_norm = nullptr;
    OF_CUDA_CHECK(cudaMallocHost(&h_total_norm, sizeof(T)));
    OF_CUDA_CHECK(cudaMemcpy(h_total_norm, total_norm, sizeof(T), cudaMemcpyDeviceToHost));
    OF_CUDA_CHECK(cudaDeviceSynchronize());

    if (not_special) {
      h_total_norm[0] = std::pow(h_total_norm[0], 1. / norm_type);
    }
    h_total_norm[0] = max_norm / (h_total_norm[0] + 1e-6);
    OF_CUDA_CHECK(cudaMemcpy(total_norm, h_total_norm, sizeof(T), cudaMemcpyHostToDevice));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    if (h_total_norm[0] < 1.) {
      std::vector<MultiScaleMulParam<T>> mut_params;
      mut_params.resize(input_size);
      for (size_t i = 0; i < input_size; ++i) {
        //auto x = ctx->Tensor4ArgNameAndIndex("model_diff", i);
        mut_params[i].size = (ctx->Tensor4ArgNameAndIndex("model_diff", i))->shape_view().elem_cnt();
        mut_params[i].data = (ctx->Tensor4ArgNameAndIndex("model_diff", i))->mut_dptr<T>();
      }
      MultiScaleMul<device_type, T> scale_mul{};
      scale_mul(ctx->stream(), mut_params, total_norm);
    }

    OF_CUDA_CHECK(cudaFreeHost(h_total_norm));
    OF_CUDA_CHECK(cudaFree(total_norm));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

} // namespace

#define REGISTER_FUSED_CLIP_GRAD_KERNEL(device, dtype)                                            \
  REGISTER_USER_KERNEL("fused_clip_grad")                                                         \
      .SetCreateFn<FusedClipGradKernel<device, dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                       \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<dtype>::value))   \
      .SetInferTmpSizeFn(InferFusedClipGradTempStorageSize);

REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, float);

}  // namespace oneflow