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
#include "oneflow/user/kernels/fused_clip_grad.h"
#include "oneflow/user/kernels/multi_reduce_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class FusedClipGradKernel final : public user_op::OpKernel {
 public:
  FusedClipGradKernel() = default;
  ~FusedClipGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t input_size = ctx->input_size("grad");
    const float max_norm = ctx->Attr<float>("max_norm");
    const float norm_type = ctx->Attr<float>("norm_type");

    std::cout << "- " << input_size << std::endl;
    std::vector<MultiReduceParam<T>> params;
    params.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("grad", i);
      params[i].size = x->shape_view().elem_cnt();
      params[i].data = x->dptr<T>();
      std::cout << i << ":" << x->shape_view().elem_cnt() << std::endl;
    }

    T *temp = nullptr;
    T *total_norm = nullptr;
    OF_CUDA_CHECK(cudaMalloc(&temp, 512 * sizeof(T)));
    OF_CUDA_CHECK(cudaMalloc(&total_norm, sizeof(T)));

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
      AbsPow<T> func{norm_type};
      MultiReduce<device_type, T, decltype(func), BinaryAdd<T>> reduce_sum{};
      reduce_sum(ctx->stream(), func, params, GetZeroVal<T>(), total_norm, temp);
      // TODO: total_norm = pow(total_norm, 1 / norm_type)
    }

    // have a check on total_norm value
    T *h_norm = nullptr;
    OF_CUDA_CHECK(cudaMallocHost(&h_norm, sizeof(T)));
    OF_CUDA_CHECK(cudaMemcpy(h_norm, total_norm, sizeof(T), cudaMemcpyDeviceToHost));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << h_norm[0] << std::endl;
    OF_CUDA_CHECK(cudaFreeHost(h_norm));

    OF_CUDA_CHECK(cudaFree(temp));
    OF_CUDA_CHECK(cudaFree(total_norm));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_FUSED_CLIP_GRAD_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("fused_clip_grad")                                                 \
      .SetCreateFn<FusedClipGradKernel<device, dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       && (user_op::HobDataType("grad", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, float);
#endif

}  // namespace

}  // namespace oneflow