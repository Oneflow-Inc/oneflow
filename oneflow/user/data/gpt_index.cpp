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
#include "oneflow/user/data/gpt_index.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace data {

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
  // log
  std::chrono::duration<double, std::milli> elapse = std::chrono::system_clock::now() - start;
  LOG(INFO) << "Load GPT Dataset index file successed, file_path: " << index_file_path
            << ", number of documents: " << this->num_docs()
            << ", number of tokens: " << this->num_tokens() << ", elapsed time: " << elapse.count()
            << " ms";
}

}  // namespace data

}  // namespace oneflow
