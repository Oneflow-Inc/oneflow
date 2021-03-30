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
#include "oneflow/user/data/gpt_index.h"
#include "oneflow/user/data/mmap_file.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

using namespace user_op;
using namespace data;

std::vector<size_t> GetSplitDocIndices(const std::vector<int64_t>& split_sizes, int64_t split_index,
                                       size_t num_docs) {
  CHECK_LT(split_index, split_sizes.size());
  size_t total_size = 0;
  FOR_RANGE(size_t, i, 0, split_sizes.size()) { total_size += split_sizes[i]; }

  std::vector<size_t> splits;
  splits.reserve(split_sizes.size());
  std::vector<size_t> splits_offsets;
  splits_offsets.reserve(split_sizes.size() + 1);
  splits_offsets.push_back(0);
  RoundModeGuard round_guard(FE_TONEAREST);
  FOR_RANGE(size_t, i, 0, split_sizes.size()) {
    float ratio = static_cast<float>(split_sizes[i]) / total_size;
    size_t split_size = static_cast<size_t>(std::nearbyint(ratio * num_docs));
    splits.push_back(split_size);
    splits_offsets.push_back(splits_offsets[i] + split_size);
  }

  std::vector<size_t> doc_indices(splits[split_index]);
  std::iota(doc_indices.begin(), doc_indices.end(), splits_offsets[split_index]);
  return doc_indices;
}

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
  index_helper_t index_helper(hierarchy.dim_vec().data(), ndim);
  int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
  index_helper.OffsetToNdIndex(rank, nd_index);
  size_t stride = 1;
  size_t index = 0;
  for (size_t i = ndim - 1; i >= 0; --i) {
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
  GPTDataLoader(KernelInitContext* ctx) : num_shards_(1), sample_index_(0), once_loaded_(false) {
    const std::string& data_file_prefix = ctx->Attr<std::string>("data_file_prefix");
    auto gpt_index = std::make_shared<const GPTIndex>(data_file_prefix + ".idx");
    auto gpt_bin = std::make_shared<MMapFile>(data_file_prefix + ".bin");
    auto doc_indices = GetSplitDocIndices(ctx->Attr<std::vector<int64_t>>("split_sizes"),
                                          ctx->Attr<int64_t>("split_index"), gpt_index->num_docs());

    dataset_.reset(new GPTDataset(gpt_index, gpt_bin, ctx->Attr<int64_t>("seq_length"),
                                  ctx->Attr<int64_t>("num_samples"), doc_indices,
                                  ctx->Attr<bool>("shuffle"), ctx->Attr<int64_t>("random_seed")));

    seq_len_ = ctx->Attr<int64_t>("seq_length");
    batch_size_ = ctx->TensorDesc4ArgNameAndIndex("sequence", 0)->shape().At(0);
    if (ctx->parallel_ctx().parallel_num() > 1) {
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      const ParallelDistribution& paral_dist =
          ctx->ParallelDistribution4ArgNameAndIndex("sequence", 0);
      CHECK_EQ(hierarchy.NumAxes(), paral_dist.sbp_parallel_size());
      num_shards_ = GetNumShards(hierarchy, paral_dist);
      CHECK_EQ(dataset_->Size() % num_shards_, 0);
      shard_index_ = GetShardIndex(hierarchy, paral_dist, ctx->parallel_ctx().parallel_id());
      CHECK_LT(shard_index_, num_shards_);
      CHECK_EQ(batch_size_ % num_shards_, 0);
      batch_size_ /= num_shards_;
      sample_index_ = shard_index_;
    }
  }
  ~GPTDataLoader() = default;

  template<typename T>
  void Next(user_op::Tensor* tokens) {
    CHECK_EQ(tokens->shape().NumAxes(), 2);
    if (once_loaded_) { once_loaded_ = true; }
    auto* dptr = tokens->mut_dptr<T>();
    for (size_t i = 0; i < batch_size_; ++i) {
      dataset_->Get(sample_index_, dptr + i * tokens->shape().At(1));
      sample_index_ += num_shards_;
    }
  }
  bool IsOnceLoaded() const { return once_loaded_; }
  void Seek(size_t iter) { sample_index_ = shard_index_ + iter * batch_size_ * num_shards_; }

 private:
  std::unique_ptr<GPTDataset> dataset_;
  size_t seq_len_;
  size_t batch_size_;
  size_t num_shards_;
  size_t shard_index_;
  size_t sample_index_;
  bool once_loaded_;
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
    if (loader->IsOnceLoaded()) { loader->Seek(*iter_ptr); }
    user_op::Tensor* tokens_tensor = ctx->Tensor4ArgNameAndIndex("sequence", 0);
    loader->Next<T>(tokens_tensor);
    *iter_ptr += 1;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_GPT_DATA_LOADER_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("gpt_data_loader")                                           \
      .SetCreateFn<GPTDataLoaderKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                           \
                       & (user_op::HobDataType("iteration", 0) == DataType::kInt64) \
                       & (user_op::HobDataType("sequence", 0) == GetDataType<dtype>::value))

REGISTER_GPT_DATA_LOADER_KERNEL(int32_t);
REGISTER_GPT_DATA_LOADER_KERNEL(int64_t);

}  // namespace oneflow
