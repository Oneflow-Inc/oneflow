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

#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace oneflow {

namespace data {

namespace {

std::ifstream::pos_type FileSize(const std::string& filename) {
  std::ifstream stream(filename, std::ifstream::ate | std::ifstream::binary);
  return stream.tellg();
}

std::vector<size_t> GetSplitDocIndices(const std::vector<int64_t>& split_sizes, size_t split_index,
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

}  // namespace

constexpr char GPTIndex::kMagicCode[];

size_t GPTIndex::num_tokens() const {
  size_t num_tokens = 0;
  for (auto size : sizes_) { num_tokens += size; }
  return num_tokens;
}

GPTIndex::GPTIndex(const std::string& index_file_path) {
  auto start = std::chrono::system_clock::now();
  std::ifstream stream(index_file_path, std::ios::binary);
  CHECK(stream.is_open());
  // verify magic code
  char magic_code[sizeof(kMagicCode)];
  stream.read(magic_code, sizeof(kMagicCode) - 1);
  magic_code[sizeof(kMagicCode) - 1] = '\0';
  CHECK_EQ(std::strcmp(magic_code, kMagicCode), 0);
  // read version
  stream.read(reinterpret_cast<char*>(&version_), sizeof(version_));
  // read dtype
  stream.read(&dtype_code_, 1);
  // read size of sizes and doc_offsets
  uint64_t sizes_size = 0;
  stream.read(reinterpret_cast<char*>(&sizes_size), sizeof(sizes_size));
  uint64_t doc_offsets_size = 0;
  stream.read(reinterpret_cast<char*>(&doc_offsets_size), sizeof(doc_offsets_size));
  // NOTE: this check is not necessary
  CHECK_EQ(sizes_size + 1, doc_offsets_size);
  // read sizes
  sizes_.resize(sizes_size);
  stream.read(reinterpret_cast<char*>(sizes_.data()),
              sizeof(decltype(sizes_)::value_type) * sizes_.size());
  // read addresses
  addresses_.resize(sizes_size);
  stream.read(reinterpret_cast<char*>(addresses_.data()),
              sizeof(decltype(addresses_)::value_type) * addresses_.size());
  // read doc_offsets
  doc_offsets_.resize(doc_offsets_size);
  stream.read(reinterpret_cast<char*>(doc_offsets_.data()),
              sizeof(decltype(doc_offsets_)::value_type) * doc_offsets_.size());
  // check eof
  CHECK_EQ(stream.tellg(), FileSize(index_file_path));
  // log
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - start;
  LOG(INFO) << "Load GPT Dataset index file successed, file_path: " << index_file_path
            << ", number of documents: " << this->num_docs()
            << ", number of tokens: " << this->num_tokens() << ", elapsed time: " << elapse.count()
            << " ms";
}

MappedBuffer::MappedBuffer(const char* filename) : mapped_(nullptr), size_(0) {
#ifdef __linux__
  int fd = open(filename, O_RDONLY);
  CHECK(fd != -1) << "open " << filename << " failed: " << strerror(errno);

  struct stat s;
  CHECK(fstat(fd, &s) != -1) << "stat " << filename << " failed: " << strerror(errno);
  size_ = s.st_size;

  mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  CHECK(mapped_ != MAP_FAILED) << "mmap " << filename << " failed: " << strerror(errno);

  close(fd);
#endif
}

MappedBuffer::~MappedBuffer() {
#ifdef __linux__
  CHECK(munmap(mapped_, size_) == 0) << "munmap failed";
#endif
}

GPTDataset::GPTDataset(const std::string& data_file_prefix, size_t seq_len, size_t num_samples,
                       const std::vector<int64_t>& split_sizes, size_t split_index, bool shuffle,
                       uint32_t seed)
    : seq_len_(seq_len), num_samples_(num_samples), shuffle_(shuffle), seed_(seed), gen_(seed) {
  auto start = std::chrono::system_clock::now();
  index_ = std::make_unique<const GPTIndex>(data_file_prefix + ".idx");
  data_ = std::make_unique<MappedBuffer>((data_file_prefix + ".bin").c_str());
  tokens_per_epoch_ = index_->num_tokens();
  num_epochs_ = GetNumEpochs();
  num_complete_epochs_ = GetNumCompleteEpochs();
  InitDocIndices(split_sizes, split_index);
  InitSampleIndices();
  InitShuffleIndices();
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - start;
  LOG(INFO) << "Create GPT Dataset successed, sequence length: " << seq_len_
            << ", number of samples: " << num_samples_
            << ", total number of samples: " << this->Size()
            << ", total number of documents: " << doc_indices_.size()
            << ", number of epochs: " << num_epochs_
            << ", number of complete epochs: " << num_complete_epochs_
            << ", shuffle: " << std::boolalpha << shuffle_ << ", random_seed: " << seed_
            << ", elapsed time: " << elapse.count() << " ms";
}

size_t GPTDataset::GetNumEpochs() const {
  // num_epochs * tokens_per_epoch >= num_samples * seq_length + 1
  // +1 is because we need to retrieve seq_length + 1 token each time
  // but the last token will overlap with the first token of the next
  // sample except for the last sample.
  return static_cast<size_t>(
      std::ceil(static_cast<double>(num_samples_ * seq_len_ + 1) / tokens_per_epoch_));
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

void GPTDataset::InitDocIndices(const std::vector<int64_t>& split_sizes, size_t split_index) {
  auto epoch_doc_indices = GetSplitDocIndices(split_sizes, split_index, index_->num_docs());
  doc_indices_.reserve(epoch_doc_indices.size() * num_complete_epochs_);
  InitDocIndices(epoch_doc_indices, num_complete_epochs_);
  if (num_epochs_ != num_complete_epochs_) { InitDocIndices(epoch_doc_indices, 1); }
}

void GPTDataset::InitDocIndices(const std::vector<size_t>& epoch_doc_indices, size_t num_epochs) {
  auto start = doc_indices_.end();
  FOR_RANGE(size_t, i, 0, num_epochs) {
    doc_indices_.insert(doc_indices_.end(), epoch_doc_indices.cbegin(), epoch_doc_indices.cend());
  }
  if (shuffle_) { std::shuffle(start, doc_indices_.end(), gen_); }
}

void GPTDataset::InitSampleIndices() {
  // + 1 is because sample_indices need an `end` mark to indicate the end position of the last
  // sample, the actual total number of samples is sample_indices_.size() - 1
  size_t total_num_samples =
      static_cast<size_t>(
          std::floor(static_cast<double>(num_epochs_ * tokens_per_epoch_ - 1) / seq_len_))
      + 1;
  sample_indices_.reserve(total_num_samples);

  size_t doc_indices_idx = 0;
  size_t doc_offset = 0;
  FOR_RANGE(size_t, i, 0, total_num_samples) {
    if (doc_indices_idx >= doc_indices_.size()) { break; }
    sample_indices_.emplace_back(doc_indices_idx, doc_offset);
    // the last sample is only used as `end` mark, there is not need to care its tokens
    if (i == total_num_samples - 1) { break; }
    int remaining_tokens = seq_len_;
    while (remaining_tokens > 0) {
      CHECK_LT(doc_indices_idx, doc_indices_.size());
      size_t doc_len = index_->doc_length(doc_indices_[doc_indices_idx]);
      CHECK_LT(doc_offset, doc_len);
      doc_len -= doc_offset;
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
  // the last sample index in sample_indices_ is an `end` mark
  shuffle_indices_.resize(sample_indices_.size() - 1);
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
