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

namespace {
template<typename T, typename IndexT>
void index_add_cpu_kernel(const int64_t n, const T* input, const IndexT* index, const T* source,
                          T* output, const int64_t stride, const int64_t source_dim,
                          const int64_t delta, const float alpha) {
  const int64_t stride_source_dim = stride * source_dim;
  for (int i = 0; i < n; i++) {
    int64_t pre_index = i / stride_source_dim;
    int64_t dim_index = (i - pre_index * stride_source_dim) / stride;
    IndexT source_dim_idx = index[dim_index];
    int64_t output_index = i + (delta * pre_index + source_dim_idx - dim_index) * stride;
    output[output_index] += static_cast<T>(alpha) * source[i];
  }
}
};  // namespace

template<typename T>
class IndexAddCpuKernel final : public user_op::OpKernel {
 public:
  IndexAddCpuKernel() = default;
  ~IndexAddCpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    const user_op::Tensor* source = ctx->Tensor4ArgNameAndIndex("source", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int64_t dim = ctx->Attr<int64_t>("dim");
    const float alpha = ctx->Attr<float>("alpha");
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& source_shape = source->shape_view();
    std::vector<int64_t> input_stride(input->stride().begin(), input->stride().end());
    const int64_t stride = input_stride[dim];
    const int64_t source_dim = source_shape.At(dim);
    const int64_t delta = input_shape.At(dim) - source_dim;
    DataType index_dtype = index->data_type();
    const int32_t n = source->shape_view().elem_cnt();
    Memcpy<DeviceType::kCPU>(
        ctx->stream(), output->mut_dptr<void>(), input->dptr<void>(),
        input->shape_view().elem_cnt() * GetSizeOfDataType(input->data_type()));
    if (GetSizeOfDataType(index_dtype) == 4) {
      index_add_cpu_kernel(n, input->dptr<T>(), index->dptr<int32_t>(), source->dptr<T>(),
                           output->mut_dptr<T>(), stride, source_dim, delta, alpha);
    } else {
      index_add_cpu_kernel(n, input->dptr<T>(), index->dptr<int64_t>(), source->dptr<T>(),
                           output->mut_dptr<T>(), stride, source_dim, delta, alpha);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_INDEX_ADD_CPU_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("index_add")                                   \
      .SetCreateFn<IndexAddCpuKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("output", 0) == GetDataType<dtype>::value));

REGISTER_INDEX_ADD_CPU_KERNEL(int8_t)
REGISTER_INDEX_ADD_CPU_KERNEL(int32_t)
REGISTER_INDEX_ADD_CPU_KERNEL(float)
REGISTER_INDEX_ADD_CPU_KERNEL(double)

}  // namespace oneflow
