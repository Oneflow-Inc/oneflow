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
#include "oneflow/user/data/gpt_data_reader.h"
#include "oneflow/user/data/gpt_dataset.h"
#include "oneflow/user/data/gpt_index.h"
#include "oneflow/user/data/mmap_file.h"
#include "oneflow/user/data/distributed_dataset.h"
#include "oneflow/user/data/batch_dataset.h"
#include "oneflow/user/data/gpt_parser.h"

namespace oneflow {

namespace data {

GPTDataReader::GPTDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
  // const std::string& data_file_prefix = ctx->Attr<std::string>("data_file_prefix");
  // auto gpt_index = std::make_shared<const GPTIndex>(data_file_prefix + ".idx");
  // auto gpt_bin = std::make_shared<MMapFile>(data_file_prefix + ".bin");

  // auto doc_indices = GetSplitDocIndices(ctx->Attr<std::vector<int64_t>>("split_sizes"),
  //                                       ctx->Attr<int64_t>("split_index"),
  //                                       gpt_index->num_docs());
  // std::unique_ptr<RandomAccessDataset<TensorBuffer>> gpt_dataset = std::make_unique<GPTDataset>(
  //     gpt_index, gpt_bin, ctx->Attr<int64_t>("seq_length"), ctx->Attr<int64_t>("num_samples"),
  //     doc_indices, ctx->Attr<bool>("shuffle"), ctx->Attr<int64_t>("random_seed"));

  // size_t batch_size = ctx->TensorDesc4ArgNameAndIndex("sequence", 0)->shape().elem_cnt();
  // if (ctx->parallel_ctx().parallel_num() > 1) {
  //   const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
  //   const ParallelDistribution& parl_dist =
  //       ctx->ParallelDistribution4ArgNameAndIndex("sequence", 0);
  //   std::unique_ptr<Dataset<TensorBuffer>> distributed_gpt_dataset =
  //       std::make_unique<DistributedDataset<TensorBuffer>>(
  //           hierarchy, parl_dist, ctx->parallel_ctx().parallel_id(), std::move(gpt_dataset));
  //   size_t dist_batch_size = GetDistributedBatchSize(batch_size, hierarchy, parl_dist);
  //   loader_.reset(
  //       new BatchDataset<TensorBuffer>(dist_batch_size, std::move(distributed_gpt_dataset)));
  // } else {
  //   loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(gpt_dataset)));
  // }

  // parser_.reset(new GPTParser());
  StartLoadThread();
}

std::vector<size_t> GPTDataReader::GetSplitDocIndices(const std::vector<int64_t>& split_sizes,
                                                      int64_t split_index, size_t num_docs) const {
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

size_t GPTDataReader::GetDistributedBatchSize(size_t batch_size, const Shape& hierarchy,
                                              const ParallelDistribution& parallel_dist) const {
  size_t num_shards = 1;
  FOR_RANGE(size_t, i, 0, parallel_dist.sbp_parallel_size()) {
    const auto& sbp_parallel = parallel_dist.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      num_shards *= hierarchy.At(sbp_parallel.split_parallel().axis());
    }
  }
  CHECK_EQ(batch_size % num_shards, 0);
  return batch_size / num_shards;
}

}  // namespace data

}  // namespace oneflow
