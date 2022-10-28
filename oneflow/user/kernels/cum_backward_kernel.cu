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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
#ifdef WITH_CUDA
namespace {
template<typename T>
__global__ void CumProdBackward(const T* dy_ptr, T* dx_ptr, const T* output_ptr, const T* input_ptr,
                                const int64_t up_space, const int64_t space,
                                const int64_t down_space, const int64_t thread_num) {
  // A thread is responsible for a row along specific dimension.
  const size_t up_space_step = space * down_space;
  CUDA_1D_KERNEL_LOOP_T(size_t, i, thread_num) {
    const size_t up_space_id = i / down_space;
    const size_t down_space_id = i % down_space;
    const size_t ptr_offset = up_space_id * up_space_step + down_space_id;
    auto* dy_ptr_base = dy_ptr + ptr_offset;
    auto* dx_ptr_base = dx_ptr + ptr_offset;
    auto* input_ptr_base = input_ptr + ptr_offset;
    auto* output_ptr_base = output_ptr + ptr_offset;

    // Buffer storing number of zero element along specific dimension.
    // Use dx as tmp buffer.
    for (size_t j = 0; j < space; j++) {
      const size_t data_offset = j * down_space;
      int is_zero = input_ptr_base[data_offset] == 0 ? 1 : 0;
      dx_ptr_base[data_offset] = is_zero + (j == 0 ? 0 : dx_ptr_base[data_offset - down_space]);
    }

    // Find index of first zero in input.
    size_t first_zero_index = space;
    for (size_t j = 0; j < space; j++) {
      const size_t data_offset = j * down_space;
      if (dx_ptr_base[data_offset] == 1) {
        first_zero_index = j;
        break;
      }
    }

    // Suppose z is index of first zero element in input,
    // for element which index is less than z grad is computed as below:
    T reverse_cumsum = 0;
    for (size_t j = 0; j < first_zero_index; j++) {
      const size_t cur_index = first_zero_index - j - 1;
      const size_t data_offset = cur_index * down_space;
      reverse_cumsum += output_ptr_base[data_offset] * dy_ptr_base[data_offset];
      dx_ptr_base[data_offset] = reverse_cumsum / input_ptr_base[data_offset];
    }

    // Where index is z, its grad is computed as below:
    if (first_zero_index == space) { return; }
    T cumprod = 1;
    T cumsum = 0;
    T cumprod_before_first_zero =
        first_zero_index == 0 ? 1 : output_ptr_base[(first_zero_index - 1) * down_space];
    for (size_t j = first_zero_index; j < space; j++) {
      const size_t down_space_offset = j * down_space;
      // Recover dx_ptr default value
      if (dx_ptr_base[down_space_offset] >= 1) { dx_ptr_base[down_space_offset] = 0; }
      if (j != first_zero_index) { cumprod *= input_ptr_base[down_space_offset]; }
      cumsum += cumprod_before_first_zero * dy_ptr_base[down_space_offset] * cumprod;
    }
    dx_ptr_base[first_zero_index * down_space] = cumsum;
  }
}
}  // namespace

template<typename T>
class GpuCumProdGradKernel final : public user_op::OpKernel {
 public:
  GpuCumProdGradKernel() = default;
  ~GpuCumProdGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto elem_cnt = dy->shape_view().elem_cnt();
    if (!elem_cnt) { return; }

    const auto* output_ptr = output->dptr<T>();
    const auto* input_ptr = input->dptr<T>();
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    // Data partition: up_space|space|down_space
    auto dim = ctx->Attr<int64_t>("dim");
    const auto up_space = elem_cnt / dx->shape_view().Count(dim);
    const auto space = dx->shape_view().At(dim);
    const auto down_space = dx->shape_view().Count(dim + 1);
    const size_t thread_num = up_space * down_space;

    if (space == 1) {
      Memcpy<DeviceType::kCUDA>(ctx->stream(), dx_ptr, dy_ptr, elem_cnt * sizeof(T));
      return;
    }
    ep::CudaLaunchConfig config{};
    ctx->stream()->As<ep::CudaStream>()->InitLaunchConfigWithWaves(
        &config, thread_num, /*DefaultBlockSize*/ 256, /*max_wave*/ 1);
    CumProdBackward<<<config.grid_dim, config.block_dim, /*shared memory*/ 0,
                      ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        dy_ptr, dx_ptr, output_ptr, input_ptr, up_space, space, down_space, thread_num);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMPROD_GRAD_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("cumprod_grad")                                 \
      .SetCreateFn<GpuCumProdGradKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMPROD_GRAD_KERNEL(float)
REGISTER_CUDA_CUMPROD_GRAD_KERNEL(double)
#undef REGISTER_CUDA_CUMPROD_GRAD_KERNEL
#endif
}  // namespace oneflow
