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
#include "oneflow/user/data/gpt_dataset.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace data {

GPTDataset::GPTDataset(const std::shared_ptr<const GPTIndex>& index,
                       const std::shared_ptr<MMapFile>& data, size_t seq_len, size_t num_samples,
                       const std::vector<size_t>& doc_indices, bool shuffle, uint32_t seed)
    : index_(index),
      data_(data),
      seq_len_(seq_len),
      num_samples_(num_samples),
      shuffle_(shuffle),
      seed_(seed),
      gen_(seed) {
  auto step = std::chrono::system_clock::now();
  tokens_per_epoch_ = 0;
  for (auto doc_index : doc_indices) { tokens_per_epoch_ += index_->doc_length(doc_index); }
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "calc tokens_per_epoch_ elapsed " << elapse.count() << " ms";
  num_epochs_ = GetNumEpochs();
  elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "calc num_epochs elapsed " << elapse.count() << " ms";
  num_complete_epochs_ = GetNumCompleteEpochs();
  elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "calc num_complete_epochs elapsed " << elapse.count() << " ms";
  InitDocIndices(doc_indices);
  elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "init doc_indices elapsed " << elapse.count() << " ms";
  InitSampleIndices();
  elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "init sample_indices elapsed " << elapse.count() << " ms";
  InitShuffleIndices();
  elapse = std::chrono::system_clock::now() - step;
  LOG(ERROR) << "init shuffle_indices elapsed " << elapse.count() << " ms";
  LOG(ERROR) << "last doc index: " << doc_indices[doc_indices.size() - 1]
             << ", num_docs: " << index_->num_docs();
}

size_t GPTDataset::GetNumEpochs() const {
  // num_epochs * tokens_per_epoch >= num_samples * seq_length + 1
  // +1 is because we need to retrieve seq_length + 1 token each time
  // but the last token will overlap with the first token of the next
  // sample except for the last sample.
  return static_cast<size_t>(
      std::ceil((num_samples_ * seq_len_ + 1) / static_cast<double>(tokens_per_epoch_)));
}

size_t GPTDataset::GetNumCompleteEpochs() const {
  if (num_epochs_ == 1) { return 1; }
  size_t num_samples_per_epoch =
      static_cast<size_t>(std::floor(static_cast<double>(tokens_per_epoch_ - 1) / seq_len_));
  size_t num_samples_exclude_last_epoch = static_cast<size_t>(
      std::floor(static_cast<double>((num_epochs_ - 1) * tokens_per_epoch_ - 1) / seq_len_));
  CHECK_LE(num_samples_exclude_last_epoch, num_samples_);
  size_t last_epoch_num_samples = num_samples_ - num_samples_exclude_last_epoch;
  CHECK_LT(last_epoch_num_samples, num_samples_per_epoch);

  bool separate_last_epoch =
      last_epoch_num_samples < static_cast<size_t>(0.8f * num_samples_per_epoch);
  return separate_last_epoch ? (num_epochs_ - 1) : num_epochs_;
}

void GPTDataset::InitDocIndices(const std::vector<size_t>& doc_indices) {
  doc_indices_.reserve(doc_indices.size() * num_complete_epochs_);
  InitDocIndices(doc_indices, num_complete_epochs_);
  if (num_epochs_ != num_complete_epochs_) { InitDocIndices(doc_indices, 1); }
}

void GPTDataset::InitDocIndices(const std::vector<size_t>& doc_indices, size_t num_epochs) {
  auto start = doc_indices_.end();
  FOR_RANGE(size_t, i, 0, num_epochs) {
    doc_indices_.insert(doc_indices_.end(), doc_indices.cbegin(), doc_indices.cend());
  }
  if (shuffle_) { std::shuffle(start, doc_indices_.end(), gen_); }
}

void GPTDataset::InitSampleIndices() {
  size_t total_num_samples =
      static_cast<size_t>(
          std::floor(static_cast<double>(num_epochs_ * tokens_per_epoch_ - 1) / seq_len_))
      + 1;

  size_t doc_indices_idx = 0;
  size_t doc_offset = 0;
  FOR_RANGE(size_t, i, 0, total_num_samples) {
    sample_indices_.emplace_back(doc_indices_idx, doc_offset);
    if (doc_indices_idx >= doc_indices_.size()) { break; }
    size_t remaining_tokens = seq_len_;
    int32_t doc_len = index_->doc_length(doc_indices_[doc_indices_idx]);
    while (remaining_tokens > 0) {
      if (remaining_tokens < doc_len) {
        // move offset inside doc
        doc_offset += remaining_tokens;
      } else {
        // move to next doc
        doc_indices_idx += 1;
        doc_offset = 0;
      }
      remaining_tokens -= doc_len;
    }
  }
  CHECK_EQ(sample_indices_.size(), total_num_samples);
  CHECK_GE(sample_indices_.size(), num_samples_);
}

void GPTDataset::InitShuffleIndices() {
  shuffle_indices_.resize(sample_indices_.size());
  std::iota(shuffle_indices_.begin(), shuffle_indices_.end(), 0);
  if (shuffle_) {
    size_t num_samples = static_cast<size_t>(
        std::floor(static_cast<double>(num_complete_epochs_ * tokens_per_epoch_ - 1) / seq_len_));
    CHECK_LE(num_samples, shuffle_indices_.size());
    std::shuffle(shuffle_indices_.begin(), shuffle_indices_.begin() + num_samples, gen_);
    if (num_complete_epochs_ != num_epochs_) {
      std::shuffle(shuffle_indices_.begin() + num_samples, shuffle_indices_.end(), gen_);
    }
  }
}

const HashMap<char, DataType> GPTDataset::kDTypeCode2DataType = {
    {1, DataType::kUInt8},
    {2, DataType::kInt8},
    {3, DataType::kInt32},  // origin {3,
                            // DataType::kInt16}
    {4, DataType::kInt32},
    {5, DataType::kInt64},
    {6, DataType::kFloat},
    {7, DataType::kDouble},
    {8, DataType::kInt32},  // origin {8, DataType::kUInt16}
};

const HashMap<char, size_t> GPTDataset::kDTypeCode2Size = {
    {1, 1},  // DataType::kUInt8
    {2, 1},  // DataType::kInt8
    {3, 2},  // DataType::kInt16
    {4, 4},  // DataType::kInt32
    {5, 8},  // DataType::kInt64
    {6, 4},  // DataType::kFloat
    {7, 8},  // DataType::kDouble
    {8, 2},  // DataType::kUInt16
};

}  // namespace data

}  // namespace oneflow
