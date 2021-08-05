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
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

class ConsistentUniqueIdStorage final {
 public:
  ConsistentUniqueIdStorage() = default;
  ~ConsistentUniqueIdStorage() = default;

  static ConsistentUniqueIdStorage* Singleton() {
    static auto* storage = new ConsistentUniqueIdStorage();
    return storage;
  }

  Maybe<void> Emplace(int64_t id, const std::string& debug_string) {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK_LE_OR_RETURN(id2debug_string_.size(), TransportToken::MaxNumberOfThreadConsistentUId());
    for (const auto& pair : id2debug_string_) { CHECK_NE_OR_RETURN(debug_string, pair.second); }
    CHECK_OR_RETURN(id2debug_string_.emplace(id, debug_string).second);
    return Maybe<void>::Ok();
  }

  Maybe<const std::string&> DebugString(int64_t id) const {
    std::unique_lock<std::mutex> lock(mutex_);
    return MapAt(id2debug_string_, id);
  }

 private:
  mutable std::mutex mutex_;
  HashMap<int64_t, std::string> id2debug_string_;
};

std::unique_ptr<int64_t>* MutThreadLocalConsistentUniqueId() {
  static thread_local std::unique_ptr<int64_t> consistent_uid;
  return &consistent_uid;
}

}  // namespace

Maybe<void> SetThisThreadConsistentUniqueId(int64_t id, const std::string& debug_string) {
  JUST(ConsistentUniqueIdStorage::Singleton()->Emplace(id, debug_string));
  auto* ptr = MutThreadLocalConsistentUniqueId();
  CHECK_ISNULL_OR_RETURN(ptr->get());
  ptr->reset(new int64_t(id));
  return Maybe<void>::Ok();
}

Maybe<int64_t> GetThisThreadConsistentUniqueId() {
  auto* ptr = MutThreadLocalConsistentUniqueId();
  CHECK_NOTNULL_OR_RETURN(ptr->get());
  return **ptr;
}

Maybe<const std::string&> GetConsistentUniqueIdDebugString(int64_t thread_consistent_unique_id) {
  return ConsistentUniqueIdStorage::Singleton()->DebugString(thread_consistent_unique_id);
}

}  // namespace oneflow
