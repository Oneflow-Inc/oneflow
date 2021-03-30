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
#ifndef ONEFLOW_USER_DATA_GPT_INDEX_H_
#define ONEFLOW_USER_DATA_GPT_INDEX_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include <vector>

namespace oneflow {

namespace data {

class GPTIndex final {
 public:
  GPTIndex(const std::string& index_file);
  ~GPTIndex() = default;

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

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GPT_INDEX_H_
