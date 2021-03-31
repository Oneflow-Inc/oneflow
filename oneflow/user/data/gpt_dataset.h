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

namespace oneflow {

namespace data {

class MegatronGPTIndex final {
 public:
  MegatronGPTIndex(const std::string& index_file);
  ~MegatronGPTIndex() = default;

  static constexpr char kMagicCode[] = "MMIDIDX\x00\x00";

  uint64_t version() const { return version_; }
  char dtype_code() const { return dtype_code_; }
  size_t num_docs() const { return sizes_.size(); }
  size_t num_tokens() const;
  size_t doc_length(size_t doc_index) const { return sizes_.at(doc_index); }
  size_t doc_offset(size_t doc_index) const { return doc_offsets_.at(doc_index); }
  size_t address(size_t doc_index) const { return addresses_.at(doc_index); }

 private:
  uint64_t version_;
  char dtype_code_;
  std::vector<int32_t> sizes_;
  std::vector<int64_t> addresses_;
  std::vector<int64_t> doc_offsets_;
};

class MegatronGPTMappedBuffer final {
 public:
  MegatronGPTMappedBuffer(const char* filename);
  ~MegatronGPTMappedBuffer();

  const void* ptr() const { return mapped_; }
  size_t size() const { return size_; }

 private:
  void* mapped_;
  size_t size_;
};

class MegatronGPTMMapDataset final {
 public:
  MegatronGPTMMapDataset(const std::string& data_file_prefix, size_t seq_len, size_t num_samples,
             const std::vector<int64_t>& split_sizes, size_t split_index, bool shuffle,
             uint32_t seed);
  OF_DISALLOW_COPY_AND_MOVE(MegatronGPTMMapDataset);
  ~MegatronGPTMMapDataset() = default;

  template<typename T>
  void Get(size_t index, T* data) const;
  size_t Size() const { return sample_indices_.size() - 1; }

 private:
  static const HashMap<char, size_t> kDTypeCode2Size;

  size_t GetNumEpochs() const;
  size_t GetNumCompleteEpochs() const;
  void InitDocIndices(const std::vector<int64_t>& split_sizes, size_t split_index);
  void InitDocIndices(const std::vector<size_t>& doc_indices, size_t num_epochs);
  void InitSampleIndices();
  void InitShuffleIndices();
  template<typename T>
  void ReadTokens(const void* src, size_t offset, T* dst, size_t size) const;

  std::unique_ptr<const MegatronGPTIndex> index_;
  std::unique_ptr<MegatronGPTMappedBuffer> data_;
  size_t seq_len_;
  size_t num_samples_;
  bool shuffle_;
  uint32_t seed_;

  size_t tokens_per_epoch_;
  size_t num_epochs_;
  size_t num_complete_epochs_;
  std::mt19937 gen_;
  std::vector<size_t> doc_indices_;
  std::vector<std::pair<size_t, size_t>> sample_indices_;
  std::vector<size_t> shuffle_indices_;
};

template<typename T>
void MegatronGPTMMapDataset::Get(size_t index, T* data) const {
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
    ReadTokens(data_->ptr(), offset, data, num_tokens);
  } else {
    size_t total_num_tokens = 0;
    // first
    size_t part_num_tokens = (index_->doc_length(doc_index) - doc_offset);
    size_t offset = index_->address(doc_index) + doc_offset * dtype_size;
    ReadTokens(data_->ptr(), offset, data, part_num_tokens);
    data += part_num_tokens;
    total_num_tokens += part_num_tokens;
    // middle
    FOR_RANGE(size_t, i, doc_indices_idx + 1, next_doc_indices_idx) {
      size_t cur_doc_index = doc_indices_[i];
      part_num_tokens = index_->doc_length(cur_doc_index);
      ReadTokens(data_->ptr(), index_->address(cur_doc_index), data, part_num_tokens);
      data += part_num_tokens;
      total_num_tokens += part_num_tokens;
    }
    // last
    part_num_tokens = next_doc_offset + 1;
    ReadTokens(data_->ptr(), index_->address(next_doc_index), data, part_num_tokens);
    total_num_tokens += part_num_tokens;
    // check
    CHECK_EQ(total_num_tokens, num_tokens);
  }
}

template<typename T>
void MegatronGPTMMapDataset::ReadTokens(const void* src, size_t bytes_offset, T* dst, size_t size) const {
  CHECK_NOTNULL(src);
  switch (index_->dtype_code()) {
#define SWITCH_CASE_ENTRY(type_code, type)                                           \
  case type_code: {                                                                  \
    const auto* src_ptr =                                                            \
        reinterpret_cast<const type*>(static_cast<const char*>(src) + bytes_offset); \
    std::copy(src_ptr, src_ptr + size, dst);                                         \
    break;                                                                           \
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
