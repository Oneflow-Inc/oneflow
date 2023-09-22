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
#ifdef WITH_CUTLASS_EXTENSION

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void RedistributionData(int64_t n, int64_t k, const T* src, T* dst) {
  const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  for (int64_t j = global_tid_y; j < n; j += blockDim.y * gridDim.y) {
    for (int64_t i = global_tid_x * 16; i < k; i += blockDim.x * gridDim.x * 16) {
      for (int m = 0; m < 4; ++m) {
        dst[j * k + i + (m * 4)] = src[j * k + i + (m * 2)];
        dst[j * k + i + (m * 4 + 1)] = src[j * k + i + (m * 2 + 1)];
        dst[j * k + i + (m * 4 + 2)] = src[j * k + i + (8 + m * 2)];
        dst[j * k + i + (m * 4 + 3)] = src[j * k + i + (8 + m * 2 + 1)];
      }
    }
  }
}

void GetBlockDims(const int64_t col_size, int* block_dim_x, int* block_dim_y) {
  const int block_size = 128;
  if ((col_size / 4) < block_size) {
    *block_dim_x = std::ceil(static_cast<float>(col_size) / 4);
    *block_dim_y = (block_size + *block_dim_x - 1) / *block_dim_x;
  } else {
    *block_dim_x = block_size;
    *block_dim_y = 1;
  }
}

int GetNumBlocks(const int64_t num_instances, const int64_t instance_per_block) {
  int max_blocks = (num_instances + instance_per_block - 1) / instance_per_block;
  return std::min(max_blocks, kCudaMaxBlocksNum);
}

}  // namespace

template<typename T>
class RedistributeKernel final : public user_op::OpKernel {
 public:
  RedistributeKernel() = default;
  ~RedistributeKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const int n = in->shape_view().At(0);
    const int k = in->shape_view().At(1);
    int block_dim_x;
    int block_dim_y;
    GetBlockDims(k, &block_dim_x, &block_dim_y);
    dim3 block_dims = dim3(block_dim_x, block_dim_y);
    const int num_blocks = GetNumBlocks(n, block_dim_y);
    RedistributionData<T>
        <<<num_blocks, block_dims, 0, cuda_stream>>>(n, k, in->dptr<T>(), out->mut_dptr<T>());
  }
};

#define REGISTER_REDISTRIBUTE_KERNEL(cpp_type, data_type)              \
  REGISTER_USER_KERNEL("redistribute")                                 \
      .SetCreateFn<RedistributeKernel<cpp_type>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("in", 0) == data_type));

REGISTER_REDISTRIBUTE_KERNEL(int8_t, DataType::kInt8)
REGISTER_REDISTRIBUTE_KERNEL(float, DataType::kFloat)
REGISTER_REDISTRIBUTE_KERNEL(half, DataType::kFloat16)

}  // namespace oneflow

#endif  // WITH_CUTLASS_EXTENSION
