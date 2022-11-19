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
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {
template<typename T, typename IndexT>
__global__ void index_add_cuda_kernel(const int64_t n, const T* input, const IndexT* index,
                                      const T* source, T* output, const int64_t stride,
                                      const int64_t source_dim, const int64_t delta,
                                      const float alpha) {
  const int64_t stride_source_dim = stride * source_dim;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t pre_index = i / stride_source_dim;
    int64_t dim_index = i % stride_source_dim / stride;
    IndexT source_dim_idx = index[dim_index];
    int64_t output_index = i + (delta * pre_index + source_dim_idx - dim_index) * stride;
    cuda::atomic::Add(output + output_index, static_cast<T>(alpha) * source[i]);
  }
}
};  // namespace

template<typename T>
class IndexAddGpuKernel final : public user_op::OpKernel {
 public:
  IndexAddGpuKernel() = default;
  ~IndexAddGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    const user_op::Tensor* source = ctx->Tensor4ArgNameAndIndex("source", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    std::cout << "enter here" << std::endl;
    const int64_t dim = ctx->Attr<int64_t>("dim");
    std::cout << "dim: " << dim << std::endl;
    const float alpha = ctx->Attr<float>("alpha");
    std::cout << "dim: " << dim << ", alpha: " << alpha << std::endl;
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& source_shape = source->shape_view();
    std::vector<int64_t> input_stride(input->stride().begin(), input->stride().end());
    const int64_t stride = input_stride[dim];
    const int64_t source_dim = source_shape.At(dim);
    const int64_t delta = input_shape.At(dim) - source_dim;
    DataType index_dtype = index->data_type();
    const int32_t n = input->shape_view().elem_cnt();
    if (GetSizeOfDataType(index_dtype) == 4) {
      RUN_CUDA_KERNEL((index_add_cuda_kernel<T, int32_t>), ctx->stream(), n, n, input->dptr<T>(),
                      index->dptr<int32_t>(), source->dptr<T>(), output->mut_dptr<T>(), stride,
                      source_dim, delta, alpha);
    } else {
      RUN_CUDA_KERNEL((index_add_cuda_kernel<T, int64_t>), ctx->stream(), n, n, input->dptr<T>(),
                      index->dptr<int64_t>(), source->dptr<T>(), output->mut_dptr<T>(), stride,
                      source_dim, delta, alpha);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_INDEX_ADD_CUDA_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("index_add")                                    \
      .SetCreateFn<IndexAddGpuKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("output", 0) == GetDataType<dtype>::value));

REGISTER_INDEX_ADD_CUDA_KERNEL(float)
REGISTER_INDEX_ADD_CUDA_KERNEL(half)
REGISTER_INDEX_ADD_CUDA_KERNEL(double)

}  // namespace oneflow
