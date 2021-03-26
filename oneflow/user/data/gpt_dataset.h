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
#ifndef ONEFLOW_USER_DATA_GPT_DATASET_H_
#define ONEFLOW_USER_DATA_GPT_DATASET_H_

#include "oneflow/user/data/dataset.h"
#include "oneflow/user/data/gpt_index.h"
#include "oneflow/user/data/mmap_file.h"

namespace oneflow {

namespace data {

class GPTDataset final : public RandomAccessDataset<TensorBuffer> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  GPTDataset(const std::shared_ptr<const GPTIndex>& index, const std::shared_ptr<MMapFile>& data,
             size_t seq_len, size_t num_samples, const std::vector<size_t>& doc_indices,
             bool shuffle, uint32_t seed);
  OF_DISALLOW_COPY_AND_MOVE(GPTDataset);
  ~GPTDataset() = default;

  LoadTargetShdPtrVec At(int64_t index) const override;
  size_t Size() const override { return sample_indices_.size(); }

 private:
  size_t GetNumEpochs() const;
  size_t GetNumCompleteEpochs() const;
  void InitDocIndices(const std::vector<size_t>& doc_indices);
  void InitDocIndices(const std::vector<size_t>& doc_indices, size_t num_epochs);
  void InitSampleIndices();
  void ShuffleSampleIndices();
  LoadTargetShdPtr GetSample(int64_t index) const;

  // config
  std::shared_ptr<const GPTIndex> index_;
  std::shared_ptr<MMapFile> data_;
  size_t seq_len_;
  size_t num_samples_;
  bool shuffle_;
  uint32_t seed_;

  // middle state
  size_t tokens_per_epoch_;
  size_t num_epochs_;
  size_t num_complete_epochs_;
  std::mt19937 gen_;
  std::vector<size_t> doc_indices_;
  std::vector<std::pair<size_t, size_t>> sample_indices_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GPT_DATASET_H_
