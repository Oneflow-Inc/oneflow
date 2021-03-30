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

#include "oneflow/core/common/util.h"
#include "oneflow/user/data/gpt_index.h"
#include "oneflow/user/data/mmap_file.h"

namespace oneflow {

namespace data {

class GPTDataset final {
 public:
  GPTDataset(const std::shared_ptr<const GPTIndex>& index, const std::shared_ptr<MMapFile>& data,
             size_t seq_len, size_t num_samples, const std::vector<size_t>& doc_indices,
             bool shuffle, uint32_t seed);
  OF_DISALLOW_COPY_AND_MOVE(GPTDataset);
  ~GPTDataset() = default;

  template<typename T>
  void Get(size_t index, T* data) const;
  size_t Size() const { return sample_indices_.size() - 1; }
  // DataType data_type() const { return kDTypeCode2DataType.at(index_->dtype_code()); }

 private:
  static const HashMap<char, DataType> kDTypeCode2DataType;
  static const HashMap<char, size_t> kDTypeCode2Size;

  size_t GetNumEpochs() const;
  size_t GetNumCompleteEpochs() const;
  void InitDocIndices(const std::vector<size_t>& doc_indices);
  void InitDocIndices(const std::vector<size_t>& doc_indices, size_t num_epochs);
  void InitSampleIndices();
  void InitShuffleIndices();
  template<typename T>
  void ReadTokens(const void* src, T* dst, size_t size) const;

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
  std::vector<size_t> shuffle_indices_;
};

template<typename T>
void GPTDataset::Get(size_t index, T* data) const {
  CHECK_LT(index, shuffle_indices_.size());
  size_t sample_index = shuffle_indices_[index];
  CHECK_LT(sample_index, sample_indices_.size() - 1);
  const size_t doc_indices_idx = sample_indices_[sample_index].first;
  const size_t doc_offset = sample_indices_[sample_index].second;
  const size_t next_doc_indices_idx = sample_indices_[sample_index + 1].first;
  const size_t next_doc_offset = sample_indices_[sample_index + 1].second;
  CHECK_LE(doc_indices_idx, next_doc_indices_idx);
  CHECK_LT(next_doc_indices_idx, doc_indices_.size());
  const size_t doc_index = doc_indices_[doc_indices_idx];
  const size_t next_doc_index = doc_indices_[next_doc_indices_idx];
  const size_t dtype_size = kDTypeCode2Size.at(index_->dtype_code());
  const size_t num_tokens = seq_len_ + 1;
  if (doc_indices_idx == next_doc_indices_idx) {
    CHECK_EQ(num_tokens, next_doc_offset - doc_offset + 1);
    size_t offset = index_->address(doc_index) + doc_offset * dtype_size;
    const void* data_addr = data_->address(offset);
    ReadTokens(data_addr, data, num_tokens);
  } else {
    size_t total_num_tokens = 0;
    // first
    size_t part_num_tokens = (index_->doc_length(doc_index) - doc_offset);
    const void* data_addr = data_->address(index_->address(doc_index) + doc_offset * dtype_size);
    ReadTokens(data_addr, data, part_num_tokens);
    data += part_num_tokens * sizeof(T);
    total_num_tokens += part_num_tokens;
    // middle
    FOR_RANGE(size_t, i, doc_indices_idx + 1, next_doc_indices_idx) {
      size_t cur_doc_index = doc_indices_[i];
      part_num_tokens = index_->doc_length(cur_doc_index);
      data_addr = data_->address(index_->address(cur_doc_index));
      ReadTokens(data_addr, data, part_num_tokens);
      data += part_num_tokens * sizeof(T);
      total_num_tokens += part_num_tokens;
    }
    // last
    part_num_tokens = next_doc_offset + 1;
    data_addr = data_->address(index_->address(next_doc_index));
    ReadTokens(data_addr, data, part_num_tokens);
    total_num_tokens += part_num_tokens;
    // check
    CHECK_EQ(total_num_tokens, num_tokens);
  }
}

template<typename T>
void GPTDataset::ReadTokens(const void* src, T* dst, size_t size) const {
  switch (index_->dtype_code()) {
#define SWITCH_CASE_ENTRY(type_code, type)         \
  case type_code: {                                \
    auto spec_src = static_cast<const type*>(src); \
    std::copy(spec_src, spec_src + size, dst);     \
    break;                                         \
  }

    SWITCH_CASE_ENTRY(1, uint8_t)
    SWITCH_CASE_ENTRY(2, int8_t)
    SWITCH_CASE_ENTRY(3, int16_t)
    SWITCH_CASE_ENTRY(4, int32_t)
    SWITCH_CASE_ENTRY(5, int64_t)
    SWITCH_CASE_ENTRY(6, float)
    SWITCH_CASE_ENTRY(7, double)
    SWITCH_CASE_ENTRY(8, uint16_t)
#undef SWITCH_CASE_ENTRY
    default: {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GPT_DATASET_H_
