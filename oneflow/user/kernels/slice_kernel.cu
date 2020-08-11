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
#include "oneflow/user/ops/slice_util.h"

namespace oneflow {

namespace {

constexpr size_t kSliceMaxDims = 8;

struct SliceGpuParams {
  int64_t ndims;
  int64_t dims[kSliceMaxDims];
  int64_t sliced_dims[kSliceMaxDims];
  int64_t begin[kSliceMaxDims];
  int64_t end[kSliceMaxDims];
  int64_t stride[kSliceMaxDims];
};

SliceGpuParams ConstructSliceGpuParams(user_op::KernelComputeContext* ctx,
                                       const user_op::Tensor* entire,
                                       const user_op::Tensor* sliced) {
  const auto& begin_vec = ctx->Attr<std::vector<int64_t>>("begin");
  const auto& end_vec = ctx->Attr<std::vector<int64_t>>("end");
  const auto& stride_vec = ctx->Attr<std::vector<int64_t>>("stride");
  const auto& has_begin_vec = ctx->Attr<std::vector<int64_t>>("has_begin");
  const auto& has_end_vec = ctx->Attr<std::vector<int64_t>>("has_end");
  CHECK_LE(entire->shape().NumAxes(), kSliceMaxDims);
  CHECK_EQ(entire->shape().NumAxes(), sliced->shape().NumAxes());
  CHECK_EQ(entire->shape().NumAxes(), begin_vec.size());
  CHECK_EQ(entire->shape().NumAxes(), end_vec.size());
  CHECK_EQ(entire->shape().NumAxes(), stride_vec.size());
  CHECK_EQ(begin_vec.size(), has_begin_vec.size());
  CHECK_EQ(end_vec.size(), has_end_vec.size());

  SliceGpuParams params;
  std::memset(&params, 0, sizeof(SliceGpuParams));
  // collapse contiguous dims who slice defautly (slice whole dim),
  // that it can reduce params.ndims thus reduce loop numbers in cuda kernel
  bool do_slice_on_prev_axis = false;
  for (int64_t i = 0; i < entire->shape().NumAxes(); ++i) {
    int64_t begin =
        has_begin_vec[i] ? RegulateSliceIndex(begin_vec.at(i), entire->shape().At(i)) : 0;
    int64_t end = has_end_vec[i] ? RegulateSliceIndex(end_vec.at(i), entire->shape().At(i))
                                 : entire->shape().At(i);
    int64_t stride = stride_vec.at(i);
    CHECK_NE(stride, 0);
    if (stride > 0) {
      CHECK_LT(begin, end);
    } else {
      CHECK_GT(begin, end);
    }
    // default slice (slice whole dim) dim can be collapsed to prev dim
    bool do_slice_on_cur_axis = (begin != 0) || (end != entire->shape().At(i)) || (stride != 1);
    if (i != 0 && !do_slice_on_prev_axis && !do_slice_on_cur_axis) {
      int64_t cur_idx = params.ndims - 1;
      params.dims[cur_idx] *= entire->shape().At(i);
      params.sliced_dims[cur_idx] *= sliced->shape().At(i);
      params.end[cur_idx] = params.dims[cur_idx];
    } else {
      params.dims[params.ndims] = entire->shape().At(i);
      params.sliced_dims[params.ndims] = sliced->shape().At(i);
      params.begin[params.ndims] = begin;
      params.end[params.ndims] = end;
      params.stride[params.ndims] = stride;
      params.ndims += 1;
    }
    do_slice_on_prev_axis = do_slice_on_cur_axis;
  }
  return params;
}

__device__ __forceinline__ void OffsetToNdIndex(const int64_t offset, const int64_t ndims,
                                                const int64_t* dims, int64_t* indices) {
  int64_t divisor = offset;
#pragma unroll
  for (int64_t i = ndims - 1; i >= 0; --i) {
    indices[i] = divisor % dims[i];
    divisor /= dims[i];
  }
}

__device__ __forceinline__ int64_t NdIndexToOffset(const int64_t ndims, const int64_t* dims,
                                                   const int64_t* indices) {
  int64_t offset = 0;
  int64_t product = 1;
#pragma unroll
  for (int64_t i = ndims - 1; i >= 0; --i) {
    offset += indices[i] * product;
    product *= dims[i];
  }
  return offset;
}

template<typename T>
__global__ void SliceForwardGpu(const int n, SliceGpuParams params, const T* entire, T* part) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    OffsetToNdIndex(i, params.ndims, params.sliced_dims, nd_index);
#pragma unroll
    for (int64_t j = 0; j < params.ndims; ++j) {
      nd_index[j] = params.begin[j] + params.stride[j] * nd_index[j];
      assert(nd_index[j] < params.dims[j]);
    }
    int64_t offset = NdIndexToOffset(params.ndims, params.dims, nd_index);
    part[i] = entire[offset];
  }
}

template<typename T>
__global__ void SliceBackwardGpu(const int n, SliceGpuParams params, const T* part, T* entire) {
  int64_t nd_index[kSliceMaxDims];
  CUDA_1D_KERNEL_LOOP(i, n) {
    OffsetToNdIndex(i, params.ndims, params.sliced_dims, nd_index);
#pragma unroll
    for (int64_t j = 0; j < params.ndims; ++j) {
      nd_index[j] = params.begin[j] + params.stride[j] * nd_index[j];
      assert(nd_index[j] < params.dims[j]);
    }
    int64_t offset = NdIndexToOffset(params.ndims, params.dims, nd_index);
    entire[offset] = part[i];
  }
}

}  // namespace

template<typename T>
class SliceGpuKernel final : public user_op::OpKernel {
 public:
  SliceGpuKernel() = default;
  ~SliceGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto params = ConstructSliceGpuParams(ctx, input, output);
    int64_t elem_cnt = output->shape().elem_cnt();
    SliceForwardGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->device_ctx()->cuda_stream()>>>(elem_cnt, params, input->dptr<T>(),
                                                             output->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class SliceGradGpuKernel final : public user_op::OpKernel {
 public:
  SliceGradGpuKernel() = default;
  ~SliceGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t dx_byte_size = dx->shape().elem_cnt() * sizeof(T);
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx->mut_dptr<T>(), 0, dx_byte_size);
    auto params = ConstructSliceGpuParams(ctx, dx, dy);
    int64_t elem_cnt = dy->shape().elem_cnt();
    SliceBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, params, dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SLICE_GPU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("slice_v2")                                                     \
      .SetCreateFn<SliceGpuKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("slice_grad_v2")                                                \
      .SetCreateFn<SliceGradGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_SLICE_GPU_KERNEL(float)
REGISTER_SLICE_GPU_KERNEL(double)
REGISTER_SLICE_GPU_KERNEL(int32_t)
REGISTER_SLICE_GPU_KERNEL(int64_t)
REGISTER_SLICE_GPU_KERNEL(int8_t)

}  // namespace oneflow
