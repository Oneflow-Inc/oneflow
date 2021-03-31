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
#include "oneflow/user/data/gpt_dataset.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

using namespace user_op;
using namespace data;

size_t GetNumShards(const Shape& hierarchy, const ParallelDistribution& parallel_dist) {
  size_t num_shards = 1;
  FOR_RANGE(size_t, i, 0, parallel_dist.sbp_parallel_size()) {
    const auto& sbp_parallel = parallel_dist.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      num_shards *= hierarchy.At(sbp_parallel.split_parallel().axis());
    }
  }
  return num_shards;
}

size_t GetShardIndex(const Shape& hierarchy, const ParallelDistribution& parallel_dist,
                     size_t rank) {
  using index_helper_t = NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>;
  size_t ndim = hierarchy.NumAxes();
  CHECK_GT(ndim, 0);
  CHECK_LE(ndim, SHAPE_MAX_AXIS_SIZE);
  index_helper_t index_helper(hierarchy.dim_vec().data(), ndim);
  int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
  index_helper.OffsetToNdIndex(rank, nd_index);
  size_t stride = 1;
  size_t index = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    const auto& sbp_parallel = parallel_dist.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      index += nd_index[i] * stride;
      stride *= hierarchy.At(i);
    }
  }
  return index;
}

class GPTDataLoader final : public OpKernelState {
 public:
  GPTDataLoader(KernelInitContext* ctx) : num_shards_(1) {
    seq_len_ = ctx->Attr<int64_t>("seq_length");

    dataset_.reset(new MegatronGPTMMapDataset(
        ctx->Attr<std::string>("data_file_prefix"), seq_len_, ctx->Attr<int64_t>("num_samples"),
        ctx->Attr<std::vector<int64_t>>("split_sizes"), ctx->Attr<int64_t>("split_index"),
        ctx->Attr<bool>("shuffle"), ctx->Attr<int64_t>("random_seed")));

    batch_size_ = ctx->LogicalTensorDesc4ArgNameAndIndex("out", 0)->shape().At(0);
    if (ctx->parallel_ctx().parallel_num() > 1) {
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      const ParallelDistribution& paral_dist = ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      CHECK_EQ(hierarchy.NumAxes(), paral_dist.sbp_parallel_size());
      num_shards_ = GetNumShards(hierarchy, paral_dist);
      CHECK_EQ(dataset_->Size() % num_shards_, 0);
      shard_index_ = GetShardIndex(hierarchy, paral_dist, ctx->parallel_ctx().parallel_id());
      CHECK_LT(shard_index_, num_shards_);
      CHECK_EQ(batch_size_ % num_shards_, 0);
      size_t device_batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().At(0);
      CHECK_EQ(batch_size_ / num_shards_, device_batch_size);
      batch_size_ = device_batch_size;
    }
  }
  ~GPTDataLoader() = default;

  template<typename T>
  void GetBatch(size_t iter, user_op::Tensor* tokens) {
    CHECK_EQ(tokens->shape().NumAxes(), 2);
    CHECK_EQ(tokens->shape().At(0), batch_size_);
    CHECK_EQ(tokens->shape().At(1), seq_len_ + 1);
    T* dptr = tokens->mut_dptr<T>();
    for (size_t i = 0; i < batch_size_; ++i) {
      size_t sample_iter = shard_index_ + (iter + 1) * i * num_shards_;
      dataset_->GetSample(sample_iter, dptr + i * (seq_len_ + 1));
    }
  }

 private:
  std::unique_ptr<MegatronGPTMMapDataset> dataset_;
  size_t seq_len_;
  size_t batch_size_;
  size_t num_shards_;
  size_t shard_index_;
};

template<typename T>
class GPTDataLoaderKernel final : public OpKernel {
 public:
  GPTDataLoaderKernel() = default;
  ~GPTDataLoaderKernel() = default;

  std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
    std::shared_ptr<OpKernelState> reader(new GPTDataLoader(ctx));
    return reader;
  }

 private:
  void Compute(KernelComputeContext* ctx, OpKernelState* state) const override {
    auto* loader = dynamic_cast<GPTDataLoader*>(state);
    user_op::Tensor* iteration_tensor = ctx->Tensor4ArgNameAndIndex("iteration", 0);
    CHECK_EQ(iteration_tensor->shape().elem_cnt(), 1);
    int64_t* iter_ptr = iteration_tensor->mut_dptr<int64_t>();
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    loader->GetBatch<T>(*iter_ptr, out_tensor);
    *iter_ptr += 1;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_GPT_DATA_LOADER_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("megatron_gpt_mmap_data_loader")                             \
      .SetCreateFn<GPTDataLoaderKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                           \
                       & (user_op::HobDataType("iteration", 0) == DataType::kInt64) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))

REGISTER_GPT_DATA_LOADER_KERNEL(int32_t);
REGISTER_GPT_DATA_LOADER_KERNEL(int64_t);

}  // namespace oneflow
