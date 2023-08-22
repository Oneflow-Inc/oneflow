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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

namespace {

const int32_t NDIMS = 16;
struct SIZE_V {
  int32_t val[NDIMS];
};

struct VIS {
  bool val[NDIMS] = {false};
};

template<typename T>
__global__ void FlipGpuForward(const int32_t element, const int64_t total_dims,
                               const SIZE_V sizes_v, const VIS vis, SIZE_V strides_v,
                               const T* in_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, element) {
    int32_t cur_indices = i;
    int32_t rem = 0;
    int32_t dst_offset = 0;
    for (int32_t d = 0; d < total_dims; d++) {
      int32_t temp = cur_indices;
      cur_indices = cur_indices / strides_v.val[d];
      rem = temp - cur_indices * strides_v.val[d];
      dst_offset += vis.val[d] ? (sizes_v.val[d] - 1 - cur_indices) * strides_v.val[d]
                               : cur_indices * strides_v.val[d];
      cur_indices = rem;
    }
    out_dptr[i] = in_dptr[dst_offset];
  }
}

/*
Example tensor:
[[0, 1, 2, 3, 4, 5, 6, 7],
 [8, 9, 10, 11, 12, 13, 14]]

Given parameters: BlockSize=4, GridSize=4
For each block_i, `block_begin_idx` is calculated as (i - 1) * BlockSize = (i - 1) * 4,
and `thread_end_idx` is set to 4 for all blocks except the final block.
In the final block, `thread_end_idx` is 2, representing the border index of the active thread.

`i_ori` is an index referring to the original position of data stored in shm[threadIdx.x] before
flipping. For instance, consider block 1 and thread 2 (element 6). The element is located at row 0,
column 7 in the tensor. Its original index `i_ori` is 7, and after flipping, it is mapped to row 0,
column 0.

                    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
global mem before:  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ A │ B │ C │ D │ x │ x │
                    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

                         block0     │    block1     │    block2     │    block3
                    ┌───┬───┬───┬───┼───┬───┬───┬───┼───┬───┬───┬───┼───┬───┬───┬───┐
shm after loading:  │ 3 │ 2 │ 1 │ 0 │ 7 │ 6 │ 5 │ 4 │ B │ A │ 9 │ 8 │ D │ C │ x │ x │
                    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

                    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
global mem after:   │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │ D │ C │ B │ A │ 9 │ 8 │ 7 │ x │ x │
                    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
*/
template<typename T>
__global__ void FlipLastDimGpuForward(const int32_t element, const int64_t last_dim_size,
                                      const T* in_dptr, T* out_dptr) {
  __shared__ T shm[ep::CudaStream::kDefaultBlockSize];
  CUDA_1D_KERNEL_LOOP(i, element) {
    int32_t block_begin_idx = blockDim.x * blockIdx.x;
    int32_t thread_end_idx = min(block_begin_idx + blockDim.x, element) - block_begin_idx;
    int32_t i_ori = block_begin_idx + (thread_end_idx - threadIdx.x - 1);
    shm[threadIdx.x] = in_dptr[i_ori];
    __syncthreads();
    int32_t row = i_ori / last_dim_size;
    int32_t col = last_dim_size - (i_ori - row * last_dim_size) - 1;
    out_dptr[row * last_dim_size + col] = shm[threadIdx.x];
  }
}

}  // namespace

template<typename T>
class FlipGpuKernel final : public user_op::OpKernel {
 public:
  FlipGpuKernel() = default;
  ~FlipGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = y_tensor->shape_view().elem_cnt();
    if (elem_cnt == 0) { return; }
    const int32_t total_dims = y_tensor->shape_view().NumAxes();

    std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
    VIS vis;
    for (auto x : dims) { vis.val[x] = true; }

    if (dims.size() == 1 && dims[0] == x_tensor->shape_view().NumAxes() - 1) {
      RUN_CUDA_KERNEL((FlipLastDimGpuForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      x_tensor->shape_view().At(total_dims - 1), x_tensor->dptr<T>(),
                      y_tensor->mut_dptr<T>());
      return;
    }

    SIZE_V sizes_v;
    for (int32_t i = 0; i < total_dims; i++) { sizes_v.val[i] = y_tensor->shape_view().At(i); }

    SIZE_V strides_v;
    for (int32_t i = 0; i < total_dims; i++) {
      strides_v.val[i] = CHECK_JUST(VectorAt(y_tensor->stride(), i));
    }
    RUN_CUDA_KERNEL((FlipGpuForward<T>), ctx->stream(), elem_cnt, elem_cnt, total_dims, sizes_v,
                    vis, strides_v, x_tensor->dptr<T>(), y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FLIP_CUDA_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("flip").SetCreateFn<FlipGpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                               \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FLIP_CUDA_KERNEL(bool)
REGISTER_FLIP_CUDA_KERNEL(float)
REGISTER_FLIP_CUDA_KERNEL(half)
REGISTER_FLIP_CUDA_KERNEL(double)
REGISTER_FLIP_CUDA_KERNEL(uint8_t)
REGISTER_FLIP_CUDA_KERNEL(int8_t)
REGISTER_FLIP_CUDA_KERNEL(int32_t)
REGISTER_FLIP_CUDA_KERNEL(int64_t)

}  // namespace oneflow
