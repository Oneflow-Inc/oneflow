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

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/embedding_kernel_util.h"

namespace oneflow {

template<typename T, typename IndexType>
class GpuEmbeddingRenormKernel final : public user_op::OpKernel {
 public:
  GpuEmbeddingRenormKernel() = default;
  ~GpuEmbeddingRenormKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const double max_norm = ctx->Attr<double>("max_norm");
    const double norm_type = ctx->Attr<double>("norm_type");

    const ShapeView& in_shape = in->shape();
    const int32_t emb_size = in_shape.At(0);
    const int32_t emb_dim = in_shape.At(1);
    const T* in_buf = in->dptr<T>();
    const IndexType* indices_buf = indices->dptr<IndexType>();
    T* out_buf = out->mut_dptr<T>();
    const int32_t num_indices = indices->shape().elem_cnt();
    int32_t* tmp_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr<int32_t>();
    Memset<DeviceType::kCUDA>(ctx->stream(), tmp_buf, 0,
                              GetCudaAlignedSize(sizeof(int32_t) * emb_size));
    EmbeddingReNormFunctor<DeviceType::kCUDA, T, IndexType>()(
        ctx->stream(), in_buf, indices_buf, out_buf, max_norm, norm_type, num_indices, emb_size,
        emb_dim, tmp_buf);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename IndexType>
class GpuEmbeddingKernel final : public user_op::OpKernel {
 public:
  GpuEmbeddingKernel() = default;
  ~GpuEmbeddingKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t padding_idx = ctx->Attr<int32_t>("padding_idx");
    const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");

    const ShapeView& out_shape = out->shape();
    const int32_t num_indices = out_shape.Count(0, out_shape.NumAxes() - 1);
    const int32_t emb_size = weight->shape().At(0);
    const int32_t emb_dim = out_shape.At(out_shape.NumAxes() - 1);
    const T* weight_buf = weight->dptr<T>();
    const IndexType* indices_buf = indices->dptr<IndexType>();
    T* out_buf = out->mut_dptr<T>();

    EmbeddingFunctor<DeviceType::kCUDA, T, IndexType>()(ctx->stream(), weight_buf, indices_buf,
                                                        out_buf, padding_idx, scale_grad_by_freq,
                                                        num_indices, emb_size, emb_dim);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename IndexType>
class GpuEmbeddingGradKernel final : public user_op::OpKernel {
 public:
  GpuEmbeddingGradKernel() = default;
  ~GpuEmbeddingGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t padding_idx = ctx->Attr<int32_t>("padding_idx");
    const bool scale_grad_by_freq = ctx->Attr<bool>("scale_grad_by_freq");

    const ShapeView& dy_shape = dy->shape();
    const int32_t num_indices = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    const int32_t emb_size = weight->shape().At(0);
    const int32_t emb_dim = dy_shape.At(dy_shape.NumAxes() - 1);

    const T* dy_buf = dy->dptr<T>();
    const IndexType* indices_buf = indices->dptr<IndexType>();
    T* dx_buf = dx->mut_dptr<T>();
    int32_t* tmp_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr<int32_t>();
    Memset<DeviceType::kCUDA>(ctx->stream(), dx_buf, 0, dx->shape().Count(0) * sizeof(T));
    Memset<DeviceType::kCUDA>(ctx->stream(), tmp_buf, 0,
                              GetCudaAlignedSize(sizeof(int32_t) * emb_size));
    EmbeddingGradFunctor<DeviceType::kCUDA, T, IndexType>()(
        ctx->stream(), dy_buf, indices_buf, dx_buf, padding_idx, scale_grad_by_freq, num_indices,
        emb_size, emb_dim, tmp_buf);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDAEMBEDDING_KERNEL(in_type, indices_type)                                       \
  REGISTER_USER_KERNEL("embedding_renorm")                                                         \
      .SetCreateFn<                                                                                \
          GpuEmbeddingRenormKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>()   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(in_type))            \
                       && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                                \
        const Shape& in_shape = ctx->InputShape("in", 0);                                          \
        const int32_t emb_size = in_shape.At(0);                                                   \
        return GetCudaAlignedSize(sizeof(int32_t) * emb_size);                                     \
      });                                                                                          \
  REGISTER_USER_KERNEL("embedding")                                                                \
      .SetCreateFn<                                                                                \
          GpuEmbeddingKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>()         \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                          \
          && (user_op::HobDataType("weight", 0) == OF_PP_PAIR_SECOND(in_type))                     \
          && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type)));             \
  REGISTER_USER_KERNEL("embedding_grad")                                                           \
      .SetCreateFn<                                                                                \
          GpuEmbeddingGradKernel<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("weight", 0) == OF_PP_PAIR_SECOND(in_type))        \
                       && (user_op::HobDataType("indices", 0) == OF_PP_PAIR_SECOND(indices_type))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                                \
        const Shape& in_shape = ctx->InputShape("weight", 0);                                      \
        const int32_t emb_size = in_shape.At(0);                                                   \
        return GetCudaAlignedSize(sizeof(int32_t) * emb_size);                                     \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDAEMBEDDING_KERNEL, EMBEDDING_DATA_TYPE_SEQ_CUDA,
                                 INDEX_DATA_TYPE_SEQ)
#undef REGISTER_CUDAEMBEDDING_KERNEL

}  // namespace oneflow
