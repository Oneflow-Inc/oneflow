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
#include "oneflow/core/thread/thread_unique_tag.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

std::unique_ptr<const std::string>* MutThreadLocalUniqueTag() {
  static thread_local std::unique_ptr<const std::string> thread_tag;
  return &thread_tag;
}

}  // namespace

Maybe<void> SetThisThreadUniqueTag(const std::string& tag) {
  auto* thread_tag = MutThreadLocalUniqueTag();
  if (*thread_tag) {
    CHECK_EQ_OR_RETURN(**thread_tag, tag) << "thread unique tag could not be reset.";
    return Maybe<void>::Ok();
  }
  static HashSet<std::string> existed_thread_tags;
  static std::mutex mutex;
  {
    std::unique_lock<std::mutex> lock(mutex);
    CHECK_OR_RETURN(existed_thread_tags.emplace(tag).second) << "duplicate thread tag found.";
  }
  thread_tag->reset(new std::string(tag));
  return Maybe<void>::Ok();
}

Maybe<const std::string&> GetThisThreadUniqueTag() {
  auto* thread_tag = MutThreadLocalUniqueTag();
  CHECK_NOTNULL_OR_RETURN(*thread_tag);
  return **thread_tag;
}

}  // namespace oneflow
