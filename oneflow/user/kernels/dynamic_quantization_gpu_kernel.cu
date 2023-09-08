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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/quantization_utils.cuh"

namespace oneflow {

template<typename T>
class GpuDynamicQuantizationKernel final : public user_op::OpKernel {
 public:
  GpuDynamicQuantizationKernel() = default;
  ~GpuDynamicQuantizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor* zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    CHECK(quantization_scheme == "affine");

    const int64_t elements = in->shape_view().elem_cnt();

    constexpr int pack_size = cuda::elementwise::PackSize<T>();
    int64_t pack_num = (elements + pack_size - 1) / pack_size;
    int grid_size = 0;
    cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
    grid_size = grid_size > 2048 ? 2048 : grid_size;

    size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), grid_size * element_bytes * 2);

    T* min_max = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    auto stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (per_layer_quantization) {
      quantization::ReduceMinMaxPerTensor<pack_size, T>
          <<<grid_size, cuda::elementwise::kBlockSize,
             cuda::elementwise::kBlockSize * element_bytes * 2, stream>>>(elements, in->dptr<T>(),
                                                                          min_max);
    } else {
      UNIMPLEMENTED() << "dynamic_quantization does not support per-channel quantization";
    }

    if (quantization_formula == "oneflow") {
      if (quantization_bit == 8) {
        quantization::ApplyDynamicQuantization<T, int8_t>(
            stream, grid_size, min_max, elements, in->dptr<T>(), quantization_bit,
            out->mut_dptr<int8_t>(), scale->mut_dptr<float>(), zero_point->mut_dptr<int8_t>());
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED() << "dynamic_quantization only support oneflow quantization formula";
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DYNAMIC_QUANTIZATION_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("dynamic_quantization")                                          \
      .SetCreateFn<GpuDynamicQuantizationKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_DYNAMIC_QUANTIZATION_KERNEL(float);
REGISTER_DYNAMIC_QUANTIZATION_KERNEL(double);
REGISTER_DYNAMIC_QUANTIZATION_KERNEL(half);

}  // namespace oneflow
