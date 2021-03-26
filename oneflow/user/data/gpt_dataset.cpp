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
  tokens_per_epoch_ = 0;
  for (auto doc_index : doc_indices) { tokens_per_epoch_ += index_->doc_length(doc_index); }
  num_epochs_ = GetNumEpochs();
  num_complete_epochs_ = GetNumCompleteEpochs();
  InitDocIndices(doc_indices);
  InitSampleIndices();
  if (shuffle_) { ShuffleSampleIndices(); }
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
  // doc_indices_.clear();
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

  // sample_indices_.clear();
  size_t doc_indices_idx = 0;
  size_t doc_offset = 0;
  FOR_RANGE(size_t, i, 0, total_num_samples) {
    sample_indices_.emplace_back(doc_indices_[doc_indices_idx], doc_offset);
    if (doc_indices_idx >= doc_indices_.size()) { break; }
    int32_t remaining_seq_len = seq_len_ + 1;
    int32_t doc_len = index_->doc_length(doc_indices_[doc_indices_idx]);
    while (remaining_seq_len > 0) {
      if (remaining_seq_len < doc_len) {
        // move offset inside doc
        doc_offset += remaining_seq_len;
      } else {
        // move to next doc
        doc_indices_idx += 1;
        doc_offset = 0;
      }
      remaining_seq_len -= doc_len;
    }
  }
  CHECK_EQ(sample_indices_.size(), total_num_samples);
  CHECK_GE(sample_indices_.size(), num_samples_);
}

void GPTDataset::ShuffleSampleIndices() {
  size_t num_samples = static_cast<size_t>(
      std::floor(static_cast<double>(num_complete_epochs_ * tokens_per_epoch_ - 1) / seq_len_));
  CHECK_LE(num_samples, sample_indices_.size());
  std::shuffle(sample_indices_.begin(), sample_indices_.begin() + num_samples, gen_);
  if (num_complete_epochs_ != num_epochs_) {
    std::shuffle(sample_indices_.begin() + num_samples, sample_indices_.end(), gen_);
  }
}

GPTDataset::LoadTargetShdPtr GPTDataset::GetSample(int64_t index) const {
  CHECK_LT(index, sample_indices_.size());
  auto sample_ptr = std::make_shared<TensorBuffer>();
  sample_ptr->Resize(Shape({static_cast<int64_t>(seq_len_ + 1)}), index_->data_type());
  const size_t cur_doc_index = sample_indices_[index].first;
  const size_t cur_doc_offset = sample_indices_[index].second;
  const size_t next_doc_index = sample_indices_[index + 1].first;
  const size_t next_doc_offset = sample_indices_[index + 1].second;
  const size_t dtype_size = GetSizeOfDataType(index_->data_type());
  if (cur_doc_index == next_doc_index) {
    // within the same document, just extract the chunk.
    CHECK_EQ(sample_ptr->elem_cnt(), next_doc_offset - cur_doc_offset + 1);
    size_t offset = index_->address(cur_doc_index) + cur_doc_offset * dtype_size;
    data_->read(sample_ptr->mut_data(), offset, sample_ptr->elem_cnt());
  } else {
    char* dptr = sample_ptr->mut_data<char>();
    size_t total_length = 0;
    size_t length = (index_->doc_length(cur_doc_index) - cur_doc_offset) * dtype_size;
    data_->read(dptr, index_->address(cur_doc_index), length);
    dptr += length;
    total_length += length;
    FOR_RANGE(size_t, i, cur_doc_index + 1, next_doc_index) {
      length = index_->doc_length(i);
      data_->read(dptr, index_->address(i), length);
      dptr += length;
      total_length += length;
    }
    length = (next_doc_offset + 1) * dtype_size;
    data_->read(dptr, index_->address(next_doc_index), length);
    total_length += length;
    CHECK_EQ(sample_ptr->nbytes(), length);
  }
  return sample_ptr;
}

GPTDataset::LoadTargetShdPtrVec GPTDataset::At(int64_t index) const {
  LoadTargetShdPtrVec ret;
  ret.push_back(GetSample(index));
  return ret;
}

}  // namespace data

}  // namespace oneflow
