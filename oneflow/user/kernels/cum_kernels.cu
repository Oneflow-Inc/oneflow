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
#include "oneflow/user/kernels/cum_kernel.h"

namespace oneflow {

namespace {

// total thread number: cs_up_space * cs_down_space
// in cs_down_space part, use cs_down_space threads
// to calculate as follows(m=cs_down_space-1, n=cs_space-1, '|' stands for dependency):
// dm0, ..., d10, d00
//  |         |    |
// dm1, ..., d11, d01
//  |         |    |
// dm2, ..., d12, d02
//  |         |    |
// ...       ...  ...
//  |         |    |
// dmn, ..., d1n, d0n
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpu(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
                                 int64_t cs_down_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_up_space * cs_down_space) {
    auto cs_up_space_id = i / cs_down_space;
    auto cs_down_space_id = i - (i / cs_down_space) * cs_down_space;

    auto* in_ptr_base = in_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;
    auto* out_ptr_base = out_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      out_ptr_base[idx] = in_ptr_base[idx];
      if (j != 0) { BinaryFunc<T>()(&out_ptr_base[idx], &out_ptr_base[idx - cs_down_space]); }
    }
  }
}
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpuUpSpaceIs1(const T* in_ptr, T* out_ptr, int64_t cs_space,
                                           int64_t cs_down_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_down_space) {
    auto* in_ptr_base = in_ptr + i;
    auto* out_ptr_base = out_ptr + i;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      out_ptr_base[idx] = in_ptr_base[idx];
      if (j != 0) { BinaryFunc<T>()(&out_ptr_base[idx], &out_ptr_base[idx - cs_down_space]); }
    }
  }
}
template<typename T, template<typename> class BinaryFunc>
__global__ void CumsumForwardGpuDownSpaceIs1(const T* in_ptr, T* out_ptr, int64_t cs_up_space,
                                             int64_t cs_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_up_space) {
    auto* in_ptr_base = in_ptr + i * cs_space;
    auto* out_ptr_base = out_ptr + i * cs_space;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      out_ptr_base[j] = in_ptr_base[j];
      if (j != 0) { BinaryFunc<T>()(&out_ptr_base[j], &out_ptr_base[j - 1]); }
    }
  }
}

// total thread number: cs_up_space * cs_down_space
// in cs_down_space part, use cs_down_space threads
// to calculate as follows(m=cs_down_space-1, n=cs_space-1, there is no dependency in backward):
// dm0, ..., d10, d00
// dm1, ..., d11, d01
// dm2, ..., d12, d02
// ...       ...  ...
// dmn, ..., d1n, d0n
template<typename T>
__global__ void CumsumBackwardGpu(const T* in_ptr, T* out_ptr, int64_t cs_space,
                                  int64_t cs_down_space, int64_t elem_cnt) {
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < elem_cnt;
       i += step) {
    auto tmp = cs_space * cs_down_space;
    auto cs_space_id = (i - (i / tmp) * tmp) / cs_down_space;
    out_ptr[i] = (cs_space - cs_space_id) * in_ptr[i];
  }
}
template<typename T>
__global__ void CumsumBackwardGpu_DownSpaceIs1(const T* in_ptr, T* out_ptr, int64_t cs_up_space,
                                               int64_t cs_space, int64_t elem_cnt) {
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < elem_cnt;
       i += step) {
    auto cs_space_id = i - (i / cs_space) * cs_space;
    out_ptr[i] = (cs_space - cs_space_id) * in_ptr[i];
  }
}

}  // namespace

template<typename T, template<typename> class BinaryFunc>
class GpuCumKernel final : public user_op::OpKernel {
 public:
  GpuCumKernel() = default;
  ~GpuCumKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* in_ptr = in->dptr<T>();
    auto* out_ptr = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim + 1);
    auto thread_num = cs_up_space * cs_down_space;

    if (cs_space == 1) { return; }

    if (cs_up_space == 1) {
      RUN_CUDA_KERNEL((CumsumForwardGpuUpSpaceIs1<T, BinaryFunc>), ctx->stream(), thread_num,
                      in_ptr, out_ptr, cs_space, cs_down_space);
    } else if (cs_down_space == 1) {
      RUN_CUDA_KERNEL((CumsumForwardGpuDownSpaceIs1<T, BinaryFunc>), ctx->stream(), thread_num,
                      in_ptr, out_ptr, cs_up_space, cs_space);
    } else {
      RUN_CUDA_KERNEL((CumsumForwardGpu<T, BinaryFunc>), ctx->stream(), thread_num, in_ptr, out_ptr,
                      cs_up_space, cs_space, cs_down_space);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMSUM_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<GpuCumKernel<dtype, BinaryAdd>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                           \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_KERNEL(int64_t)
REGISTER_CUDA_CUMSUM_KERNEL(float)
REGISTER_CUDA_CUMSUM_KERNEL(double)
#undef REGISTER_CUDA_CUMSUM_KERNEL

#define REGISTER_CUDA_CUMPROD_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("cumprod").SetCreateFn<GpuCumKernel<dtype, BinaryProd>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                             \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMPROD_KERNEL(int64_t)
REGISTER_CUDA_CUMPROD_KERNEL(float)
REGISTER_CUDA_CUMPROD_KERNEL(double)
#undef REGISTER_CUDA_CUMPROD_KERNEL

template<typename T>
class GpuCumsumGradKernel final : public user_op::OpKernel {
 public:
  GpuCumsumGradKernel() = default;
  ~GpuCumsumGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // judge whether tensor has 0 size dimension first
    const auto* in = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto elem_cnt = in->shape().elem_cnt();
    if (!elem_cnt) { return; }
    auto* out = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* in_ptr = in->dptr<T>();
    auto* out_ptr = out->mut_dptr<T>();

    // take cumsum's abbreviation as `cs`
    // data partition: cs_up_space|cs_space|cs_down_space
    auto cs_up_space = elem_cnt / in->shape().Count(dim);
    auto cs_space = in->shape().At(dim);
    auto cs_down_space = in->shape().Count(dim + 1);
    auto thread_num = elem_cnt;

    if (cs_down_space == 1) {
      RUN_CUDA_KERNEL((CumsumBackwardGpu_DownSpaceIs1<T>), ctx->stream(), thread_num, in_ptr,
                      out_ptr, cs_up_space, cs_space, elem_cnt);
    } else {
      RUN_CUDA_KERNEL((CumsumBackwardGpu<T>), ctx->stream(), thread_num, in_ptr, out_ptr, cs_space,
                      cs_down_space, elem_cnt);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_CUMSUM_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("cumsum_grad")                                  \
      .SetCreateFn<GpuCumsumGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_CUMSUM_GRAD_KERNEL(float)
REGISTER_CUDA_CUMSUM_GRAD_KERNEL(double)
#undef REGISTER_CUDA_CUMSUM_GRAD_KERNEL

namespace {
template<typename T>
__global__ void cumprod_backward(const T* dy_ptr, T* dx_ptr, const T* output_ptr,
                                 const T* input_ptr, const int64_t up_space, const int64_t space,
                                 const int64_t down_space, const int64_t thread_num) {
  extern __shared__ size_t block_cumsum_zero_number[];
  // a thread compute along the specific dim
  const size_t step = space * down_space;
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < thread_num;
       i += gridDim.x * blockDim.x) {
    size_t* cumsum_zero_number = block_cumsum_zero_number + threadIdx.x * space;
    const size_t up_space_id = i / down_space;
    const size_t down_space_id = i % down_space;
    const size_t ptr_offset = up_space_id * step + down_space_id;
    auto* dy_ptr_base = dy_ptr + ptr_offset;
    auto* dx_ptr_base = dx_ptr + ptr_offset;
    auto* input_ptr_base = input_ptr + ptr_offset;
    auto* output_ptr_base = output_ptr + ptr_offset;

    // buffer for a row
    for (size_t j = 0; j < space; j++) {
      int is_zero = input_ptr_base[j * down_space] == 0 ? 1 : 0;
      cumsum_zero_number[j] = is_zero + (j == 0 ? 0 : cumsum_zero_number[j - 1]);
    }

    // for k < z(z is first zero index)
    T reverse_cumsum = 0;
    for (size_t j = 0; j < space; j++) {
      const size_t cur_index = space - j - 1;
      const size_t data_offset = cur_index * down_space;
      if (cumsum_zero_number[cur_index] > 0) { continue; }
      reverse_cumsum += output_ptr_base[data_offset];
      dx_ptr_base[data_offset] = reverse_cumsum / input_ptr_base[data_offset];
    }

    // for k == z
    size_t first_zero_index = space;
    for (size_t j = 0; j < space; j++) {
      if (cumsum_zero_number[j] == 1) {
        first_zero_index = j;
        break;
      }
    }
    if (first_zero_index == space) { return; }
    T cumprod = 1;
    T cumsum = 0;
    T cumprod_before_first_zero =
        first_zero_index == 0 ? 1 : output_ptr_base[(first_zero_index - 1) * down_space];
    for (size_t j = first_zero_index; j < space; j++) {
      if (cumsum_zero_number[j] != 1) { continue; }
      const size_t down_space_offset = j * down_space;
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
    auto elem_cnt = dy->shape().elem_cnt();
    if (!elem_cnt) { return; }

    const auto* output_ptr = output->dptr<T>();
    const auto* input_ptr = input->dptr<T>();
    const auto* dy_ptr = dy->dptr<T>();
    auto* dx_ptr = dx->mut_dptr<T>();

    auto dim = ctx->Attr<int64_t>("dim");
    auto up_space = elem_cnt / dx->shape().Count(dim);
    auto space = dx->shape().At(dim);
    auto down_space = dx->shape().Count(dim + 1);
    size_t thread_num = up_space * down_space;

    if (space == 1) { return; }
    ep::CudaLaunchConfig config{};
    ctx->stream()->As<ep::CudaStream>()->InitLaunchConfigWithWaves(
        &config, thread_num, /*DefaultBlockSize*/ 256, /*max_wave*/ 1);
    const size_t shm_byte_size =
        std::min((size_t)config.block_dim.x, thread_num) * space * sizeof(size_t);
    cumprod_backward<<<config.grid_dim, config.block_dim, shm_byte_size,
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
}  // namespace oneflow
