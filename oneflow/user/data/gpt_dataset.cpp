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

#ifdef __linux__
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

namespace oneflow {

namespace data {

namespace {

void GetSplitDocIndices(std::vector<size_t>* doc_indices, const std::vector<int64_t>& split_sizes,
                        size_t split_index, size_t num_docs) {
  CHECK_LT(split_index, split_sizes.size());
  size_t total_size = 0;
  FOR_RANGE(size_t, i, 0, split_sizes.size()) { total_size += split_sizes[i]; }

  size_t split_offset = 0;
  RoundModeGuard round_guard(FE_TONEAREST);
  FOR_RANGE(size_t, i, 0, split_index) {
    float ratio = static_cast<float>(split_sizes[i]) / total_size;
    size_t split_size = static_cast<size_t>(std::nearbyint(ratio * num_docs));
    split_offset += split_size;
  }

  float ratio = static_cast<float>(split_sizes[split_index]) / total_size;
  size_t split_size = static_cast<size_t>(std::nearbyint(ratio * num_docs));
  doc_indices->resize(split_size);
  std::iota(doc_indices->begin(), doc_indices->end(), split_offset);
}

}  // namespace

constexpr char MegatronGPTIndex::kMagicCode[];

MegatronGPTIndex::MegatronGPTIndex(const std::string& index_file_path) {
  auto start = std::chrono::system_clock::now();
  std::ifstream stream(index_file_path, std::ios::binary);
  CHECK(stream.is_open());
  // verify magic code
  char magic_code[sizeof(kMagicCode)];
  stream.read(magic_code, sizeof(kMagicCode) - 1);
  magic_code[sizeof(kMagicCode) - 1] = '\0';
  CHECK_EQ(std::strncmp(magic_code, kMagicCode, sizeof(kMagicCode)), 0);
  // read version
  stream.read(reinterpret_cast<char*>(&version_), sizeof(version_));
  // read dtype
  stream.read(&dtype_code_, sizeof(dtype_code_));
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
  int pos = stream.tellg();
  stream.seekg(0, std::ios_base::end);
  CHECK_EQ(pos, stream.tellg());
  // log
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - start;
  LOG(INFO) << "Load GPT Dataset index file successed, file_path: " << index_file_path
            << ", number of documents: " << this->num_docs() << ", elapsed time: " << elapse.count()
            << " ms";
}

MappedBuffer::MappedBuffer(const std::string& filename)
    : mapped_(nullptr), size_(0) {
#ifdef __linux__
  int fd = open(filename.c_str(), O_RDONLY);
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

MegatronGPTMMapDataset::MegatronGPTMMapDataset(const std::string& data_file_prefix, size_t seq_len,
                                               size_t label_len, size_t num_samples,
                                               const std::vector<int64_t>& split_sizes,
                                               size_t split_index, bool shuffle, uint32_t seed)
    : seq_len_(seq_len),
      sample_len_(seq_len + label_len),
      num_samples_(num_samples),
      shuffle_(shuffle),
      seed_(seed),
      gen_(seed) {
  auto start = std::chrono::system_clock::now();
  index_ = std::make_unique<const MegatronGPTIndex>(data_file_prefix + ".idx");
  data_ = std::make_unique<const MappedBuffer>(data_file_prefix + ".bin");
  dtype_size_ = kDTypeCode2Size.at(index_->dtype_code());
  std::vector<size_t> epoch_doc_indices;
  GetSplitDocIndices(&epoch_doc_indices, split_sizes, split_index, index_->num_docs());
  tokens_per_epoch_ = GetNumTokens(epoch_doc_indices);
  num_epochs_ = GetNumEpochs();
  num_complete_epochs_ = GetNumCompleteEpochs();
  InitDocIndices(epoch_doc_indices);
  InitSampleIndices();
  InitShuffleIndices();
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - start;
  LOG(INFO) << "Create GPT Dataset successed, sequence length: " << seq_len_
            << ", number of samples: " << num_samples_
            << ", total number of samples: " << shuffle_indices_.size()
            << ", total number of documents: " << doc_indices_.size()
            << ", number of epochs: " << num_epochs_
            << ", number of complete epochs: " << num_complete_epochs_
            << ", shuffle: " << std::boolalpha << shuffle_ << ", random_seed: " << seed_
            << ", elapsed time: " << elapse.count() << " ms";
}

size_t MegatronGPTMMapDataset::GetNumTokens(const std::vector<size_t>& doc_indices) const {
  size_t num_tokens = 0;
  for (auto doc_index : doc_indices) { num_tokens += index_->doc_length(doc_index); }
  return num_tokens;
}

size_t MegatronGPTMMapDataset::GetNumEpochs() const {
  // num_epochs * tokens_per_epoch >= num_samples * seq_length + 1
  // +1 is because we need to retrieve seq_length + 1 token each time
  // but the last token will overlap with the first token of the next
  // sample except for the last sample.
  return static_cast<size_t>(
      std::ceil(static_cast<double>(num_samples_ * seq_len_ + 1) / tokens_per_epoch_));
}

size_t MegatronGPTMMapDataset::GetNumCompleteEpochs() const {
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

void MegatronGPTMMapDataset::InitDocIndices(const std::vector<size_t>& epoch_doc_indices) {
  doc_indices_.reserve(epoch_doc_indices.size() * num_epochs_);
  InitDocIndices(epoch_doc_indices, num_complete_epochs_);
  if (num_epochs_ != num_complete_epochs_) { InitDocIndices(epoch_doc_indices, 1); }
}

void MegatronGPTMMapDataset::InitDocIndices(const std::vector<size_t>& epoch_doc_indices,
                                            size_t num_epochs) {
  auto start = std::distance(doc_indices_.cbegin(), doc_indices_.cend());
  FOR_RANGE(size_t, i, 0, num_epochs) {
    doc_indices_.insert(doc_indices_.end(), epoch_doc_indices.cbegin(), epoch_doc_indices.cend());
  }
  if (shuffle_) { std::shuffle(doc_indices_.begin() + start, doc_indices_.end(), gen_); }
}

void MegatronGPTMMapDataset::InitSampleIndices() {
  size_t total_num_samples = static_cast<size_t>(
      std::floor(static_cast<double>(num_epochs_ * tokens_per_epoch_ - 1) / seq_len_));
  sample_indices_.reserve(total_num_samples);

  size_t doc_indices_idx = 0;
  size_t doc_offset = 0;
  FOR_RANGE(size_t, i, 0, total_num_samples) {
    if (doc_indices_idx >= doc_indices_.size()) { break; }
    sample_indices_.emplace_back(doc_indices_idx, doc_offset);
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

void MegatronGPTMMapDataset::InitShuffleIndices() {
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

const HashMap<char, size_t> MegatronGPTMMapDataset::kDTypeCode2Size = {
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
