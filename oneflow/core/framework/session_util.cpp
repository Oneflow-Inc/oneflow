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
#include "oneflow/core/framework/session_util.h"

namespace oneflow {

namespace {

std::mutex* GlobalSessionUtilMutex() {
  static std::mutex global_id2session_map_mutex;
  return &global_id2session_map_mutex;
}

std::vector<int64_t>* RegsiteredSessionIds() {
  static std::vector<int64_t> default_sess_id;
  return &default_sess_id;
}

Maybe<void> SetDefaultSessionId(int64_t val) {
  std::vector<int64_t>* ids = RegsiteredSessionIds();
  ids->emplace_back(val);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<int64_t> GetDefaultSessionId() {
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  const auto& regsitered_ids = *(RegsiteredSessionIds());
  CHECK_GT_OR_RETURN(regsitered_ids.size(), 0);
  return regsitered_ids.back();
}

bool RegsterSessionId(int64_t session_id) {
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  auto* regsitered_ids = RegsiteredSessionIds();
  auto itor = std::find(regsitered_ids->begin(), regsitered_ids->end(), session_id);
  if (itor != regsitered_ids->end()) { return false; }
  regsitered_ids->push_back(session_id);
  return true;
}

bool ClearSessionId(int64_t session_id) {
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  auto* regsitered_ids = RegsiteredSessionIds();
  auto itor = std::find(regsitered_ids->begin(), regsitered_ids->end(), session_id);
  if (itor == regsitered_ids->end()) { return false; }
  regsitered_ids->erase(itor);
  return true;
}

}  // namespace oneflow
