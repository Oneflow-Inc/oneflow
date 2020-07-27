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
#include "oneflow/xrt/compilation_cache.h"

namespace oneflow {
namespace xrt {

bool operator==(const Signature &lhs, const Signature &rhs) {
  return lhs.builder_name == rhs.builder_name && lhs.device_ordinal == rhs.device_ordinal
         && lhs.entry_shapes == rhs.entry_shapes;
}

size_t SignatureHash::operator()(const Signature &signature) const {
  size_t hash_val =
      std::hash<std::string>()(signature.builder_name) ^ std::hash<int>()(signature.device_ordinal);
  for (const auto &shape : signature.entry_shapes) { hash_val ^= std::hash<Shape>()(shape); }
  return hash_val;
}

Signature ComputeSignature(const std::string &name, const int device_ordinal,
                           const std::vector<Parameter> &entry_params) {
  Signature signature;
  signature.builder_name = name;
  signature.device_ordinal = device_ordinal;
  signature.entry_shapes.resize(entry_params.size());
  for (int i = 0; i < entry_params.size(); ++i) {
    signature.entry_shapes[i] = entry_params[i].shape();
  }
  return std::move(signature);
}

Executable *CompilationCache::GetRecord(const Signature &signature) const {
  Executable *record = nullptr;
  // std::shared_lock<std::shared_mutex> lock(mutex_);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto &it = records_.find(signature);
  if (it != records_.end()) { record = it->second.get(); }
  return record;
}

void CompilationCache::Record(const Signature &signature,
                              const std::shared_ptr<Executable> &result) {
  // std::unique_lock<std::shared_mutex> lock(mutex_);
  std::lock_guard<std::mutex> lock(mutex_);
  records_.emplace(signature, result);
}

void CompilationCache::Release() {
  util::Map<Signature, std::shared_ptr<Executable>, SignatureHash> empty_records;
  records_.swap(empty_records);
}

}  // namespace xrt
}  // namespace oneflow
