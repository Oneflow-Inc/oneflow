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
#include "oneflow/core/embedding/embedding.cuh"
#include "oneflow/core/embedding/hash_function.cuh"
#include "oneflow/core/job/embedding.h"
#include "oneflow/user/kernels/unique_kernel_util.h"
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include "oneflow/user/kernels/model_update_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename IDX>
__global__ void SGDUpdateGpu(float weight_decay, float learning_rate_val,
                             const int64_t feature_size, const IDX* num_unique_indices,
                             const float* learning_rate, const int64_t* skip_if,
                             const T* model_diff, const T* model, T* updated_model) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  const int64_t n = *num_unique_indices * feature_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T model_val = model[i];
    updated_model[i] = model_val - learning_rate_val * (model_diff[i] + weight_decay * model_val);
  }
}

__global__ void PrintData(const int32_t* num, const int64_t* indices, const float* values) {
  int count = *num;
  CUDA_1D_KERNEL_LOOP(i, count) {
    printf("indices %d of %d: %ld values: %f\n", i, count, indices[i], values[(i + 1) * 128 - 1]);
  }
}

}  // namespace

template<typename K, typename IDX>
class EmbeddingPrefetchKernel final : public user_op::OpKernel {
 public:
  EmbeddingPrefetchKernel() = default;
  ~EmbeddingPrefetchKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(INFO) << "EmbeddingPrefetchKernel";
    const std::string& name = ctx->Attr<std::string>("name");
    const std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>>& embedding =
        Global<EmbeddingMgr>::Get()->GetEmbedding4Name(
            name);  // StrCat(name, ctx->parallel_ctx().parallel_id()));
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* num_unique_indices = ctx->Tensor4ArgNameAndIndex("num_unique_indices", 0);
    user_op::Tensor* unique_indices = ctx->Tensor4ArgNameAndIndex("unique_indices", 0);
    user_op::Tensor* reverse_idx = ctx->Tensor4ArgNameAndIndex("reverse_idx", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t unique_workspace_bytes;
    UniqueKernelUtil<DeviceType::kGPU, K, IDX>::GetUniqueWorkspaceSizeInBytes(
        ctx->device_ctx(), indices->shape().elem_cnt(), &unique_workspace_bytes);
    char* unique_workspace_ptr = tmp_buffer->mut_dptr<char>();
    // TODO: partiation indices
    UniqueKernelUtil<DeviceType::kGPU, K, IDX>::Unique(
        ctx->device_ctx(), indices->shape().elem_cnt(), indices->dptr<K>(),
        num_unique_indices->mut_dptr<IDX>(), unique_indices->mut_dptr<K>(),
        reverse_idx->mut_dptr<IDX>(), unique_workspace_ptr, unique_workspace_bytes);
    // prefetch
    cudaError_t error =
        embedding->Prefetch(num_unique_indices->dptr<IDX>(), unique_indices->dptr<K>(),
                            ctx->device_ctx()->cuda_stream());
    OF_CUDA_CHECK(error);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename K, typename IDX>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) -> size_t {
    const user_op::TensorDesc& indices = ctx->InputTensorDesc("indices", 0);
    int64_t unique_workspace_bytes;
    UniqueKernelUtil<DeviceType::kGPU, K, IDX>::GetUniqueWorkspaceSizeInBytes(
        nullptr, indices.shape().elem_cnt(), &unique_workspace_bytes);
    return unique_workspace_bytes;
  };
}

REGISTER_USER_KERNEL("embedding_prefetch")
    .SetCreateFn<EmbeddingPrefetchKernel<int64_t, int32_t>>()
    .SetIsMatchedHob(user_op::HobTrue())
    .SetInferTmpSizeFn(GenInferTmpSizeFn<int64_t, int32_t>());

template<typename T, typename K, typename IDX>
class EmbeddingLookupKernel final : public user_op::OpKernel {
 public:
  EmbeddingLookupKernel() = default;
  ~EmbeddingLookupKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(INFO) << "EmbeddingLookupKernel";
    const std::string& name = ctx->Attr<std::string>("name");
    const std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>>& embedding =
        Global<EmbeddingMgr>::Get()->GetEmbedding4Name(
            name);  // StrCat(name, ctx->parallel_ctx().parallel_id()));
    const user_op::Tensor* num_unique_indices =
        ctx->Tensor4ArgNameAndIndex("num_unique_indices", 0);
    const user_op::Tensor* unique_indices = ctx->Tensor4ArgNameAndIndex("unique_indices", 0);
    const user_op::Tensor* reverse_idx = ctx->Tensor4ArgNameAndIndex("reverse_idx", 0);
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    cudaError_t error =
        embedding->Lookup(num_unique_indices->dptr<IDX>(), unique_indices->dptr<K>(),
                          unique_values->mut_dptr<T>(), ctx->device_ctx()->cuda_stream());
    OF_CUDA_CHECK(error);
    // only support sgd, if adam slice embedding from unique_values
    GatherKernelUtilImpl<DeviceType::kGPU, T, IDX>::Forward(
        ctx->device_ctx(), reverse_idx->dptr<IDX>(), unique_indices->shape().elem_cnt(),
        unique_values->mut_dptr<T>(),
        Shape({1, unique_indices->shape().elem_cnt(), ctx->Attr<int64_t>("embedding_size")}),
        embeddings->mut_dptr<T>(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("embedding_lookup")
    .SetCreateFn<EmbeddingLookupKernel<float, int64_t, int32_t>>()
    .SetIsMatchedHob(user_op::HobTrue());

template<typename T, typename K, typename IDX>
class SGDEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  SGDEmbeddingUpdateKernel() = default;
  ~SGDEmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(INFO) << "SGDEmbeddingUpdateKernel";
    const user_op::Tensor* num_unique_indices =
        ctx->Tensor4ArgNameAndIndex("num_unique_indices", 0);
    const user_op::Tensor* unique_indices = ctx->Tensor4ArgNameAndIndex("unique_indices", 0);
    const user_op::Tensor* reverse_idx = ctx->Tensor4ArgNameAndIndex("reverse_idx", 0);
    const user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float* learning_rate_ptr = nullptr;
    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    T* unique_diff_ptr = tmp_buffer->mut_dptr<T>();
    T* unique_update_values_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                             + GetCudaAlignedSize(embedding_diff->shape().elem_cnt() * sizeof(T)));
    int64_t outer_dim_size = 1;
    int64_t num_indices = unique_indices->shape().elem_cnt();
    int64_t inner_dim_size = embedding_diff->shape().elem_cnt() / num_indices;
    int64_t num_reverse_idx = reverse_idx->shape().elem_cnt();
    const std::string& name = ctx->Attr<std::string>("name");
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const std::shared_ptr<embedding::Embedding<int64_t, float, XXH64, int32_t>>& embedding =
        Global<EmbeddingMgr>::Get()->GetEmbedding4Name(
            name);  // StrCat(name, ctx->parallel_ctx().parallel_id()));
    Memset<DeviceType::kGPU>(ctx->device_ctx(), unique_diff_ptr, 0,
                             embedding_diff->shape().elem_cnt() * sizeof(T));
    UnsortedSegmentSumKernelUtil<DeviceType::kGPU, T, IDX, T>::UnsortedSegmentSum(
        ctx->device_ctx(), reverse_idx->dptr<IDX>(), embedding_diff->dptr<T>(), num_reverse_idx,
        num_indices, outer_dim_size, inner_dim_size, 0, unique_diff_ptr);
    SGDUpdateGpu<T, IDX><<<BlocksNum4ThreadsNum(embedding_diff->shape().elem_cnt()),
                           kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
        0, 0, embedding_size, num_unique_indices->dptr<IDX>(), learning_rate_ptr, skip_if_ptr,
        unique_diff_ptr, unique_values->dptr<T>(), unique_update_values_ptr);
    cudaError_t error =
        embedding->Update(num_unique_indices->dptr<IDX>(), unique_indices->dptr<K>(),
                          unique_update_values_ptr, ctx->device_ctx()->cuda_stream());
    OF_CUDA_CHECK(error);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

template<typename T, typename K>
user_op::InferTmpSizeFn GenUpdateInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& embedding_diff = ctx->InputTensorDesc("embedding_diff", 0);
    return 2 * GetCudaAlignedSize(embedding_diff.shape().elem_cnt() * sizeof(T));
  };
}

REGISTER_USER_KERNEL("sgd_embedding_update")
    .SetCreateFn<SGDEmbeddingUpdateKernel<float, int64_t, int32_t>>()
    .SetIsMatchedHob(user_op::HobTrue())
    .SetInferTmpSizeFn(GenUpdateInferTmpSizeFn<float, int64_t>());

class AdamEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  AdamEmbeddingUpdateKernel() = default;
  ~AdamEmbeddingUpdateKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(INFO) << "AdamEmbeddingUpdateKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("adam_embedding_update")
    .SetCreateFn<AdamEmbeddingUpdateKernel>()
    .SetIsMatchedHob(user_op::HobTrue());

}  // namespace oneflow
