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
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

TeePersistentLogStream::TeePersistentLogStream(const std::string& path) {
  destinations_.emplace_back(LocalFS(), FLAGS_log_dir);
  branches_.reserve(destinations_.size());
  for (const auto& destination : destinations_) {
    branches_.emplace_back(std::make_unique<PersistentOutStream>(
        destination.mut_file_system(), JoinPath(destination.base_dir(), path)));
  }
}

TeePersistentLogStream::~TeePersistentLogStream() { Flush(); }

std::unique_ptr<TeePersistentLogStream> TeePersistentLogStream::Create(const std::string& path) {
  auto stream_ptr = new TeePersistentLogStream(path);
  return std::unique_ptr<TeePersistentLogStream>(stream_ptr);
}

void TeePersistentLogStream::Flush() {
  for (const auto& branch : branches_) { branch->Flush(); }
};

void TeePersistentLogStream::Write(const char* s, size_t n) {
  for (const auto& branch : branches_) { branch->Write(s, n); }
};

void TeePersistentLogStream::Write(const std::string& str) { this->Write(str.data(), str.size()); }

void TeePersistentLogStream::Write(const PbMessage& proto) {
  std::string output;
  google::protobuf::TextFormat::PrintToString(proto, &output);
  this->Write(output);
}

}  // namespace oneflow
