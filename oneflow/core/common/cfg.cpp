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
#include "oneflow/core/common/cfg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

bool FieldDefinedInCfgMessage(const CfgMessage& msg, const std::string& field_name) {
  return msg.FieldDefined4FieldName(field_name);
}

const CfgMessage& GetMessageInCfgMessage(const CfgMessage& msg, int field_index) {
  return *CHECK_NOTNULL(msg.FieldPtr4FieldNumber<CfgMessage>(field_index));
}

const CfgMessage& GetMessageInCfgMessage(const CfgMessage& msg, const std::string& field_name) {
  return *CHECK_NOTNULL(msg.FieldPtr4FieldName<CfgMessage>(field_name));
}

CfgMessage* MutableMessageInCfgMessage(CfgMessage* msg, const std::string& field_name) {
  return CHECK_NOTNULL(msg->MutableFieldPtr4FieldName<CfgMessage>(field_name));
}

CfgMessage* MutableMessageInCfgMessage(CfgMessage* msg, int field_index) {
  return CHECK_NOTNULL(msg->MutableFieldPtr4FieldNumber<CfgMessage>(field_index));
}

std::string GetStrValInCfgFdOrCfgRpf(const CfgMessage& msg,
                                     const std::string& fd_name_may_have_idx) {
  if (msg.FieldDefined4FieldName<std::string>(fd_name_may_have_idx)) {
    return *msg.FieldPtr4FieldName<std::string>(fd_name_may_have_idx);
  }
  std::string field_name;
  int32_t index = 0;
  GetPrefixAndIndex(fd_name_may_have_idx, &field_name, &index);
  const auto* rp_field = msg.FieldPtr4FieldName<CfgRpf<std::string>>(field_name);
  CHECK_NOTNULL(rp_field);
  CHECK_LT(index, rp_field->size());
  return rp_field->Get(index);
}

bool HasStrFieldInCfgFdOrCfgRpf(const CfgMessage& msg, const std::string& fd_name_may_have_idx) {
  if (msg.FieldDefined4FieldName<std::string>(fd_name_may_have_idx)) { return true; }
  std::string field_name;
  int32_t index = 0;
  if (!TryGetPrefixAndIndex(fd_name_may_have_idx, &field_name, &index)) { return false; }
  const auto* rp_field = msg.FieldPtr4FieldName<CfgRpf<std::string>>(field_name);
  return rp_field != nullptr && index < rp_field->size();
}

std::string ReplaceStrValInCfgFdOrCfgRpf(CfgMessage* msg, const std::string& fd_name_may_have_idx,
                                         const std::string& new_val) {
  if (msg->FieldDefined4FieldName<std::string>(fd_name_may_have_idx)) {
    // Do not define old_val with type const std::string&, because the value will be changed
    const std::string old_val = *msg->FieldPtr4FieldName<std::string>(fd_name_may_have_idx);
    *msg->MutableFieldPtr4FieldName<std::string>(fd_name_may_have_idx) = new_val;
    return old_val;
  }
  std::string field_name;
  int32_t index = 0;
  GetPrefixAndIndex(fd_name_may_have_idx, &field_name, &index);
  const auto* rp_field = msg->FieldPtr4FieldName<CfgRpf<std::string>>(field_name);
  CHECK_NOTNULL(rp_field);
  CHECK_LT(index, rp_field->size());
  // Do not define old_val with type const std::string&, because the value will be changed
  const std::string old_val = rp_field->Get(index);
  *msg->MutableFieldPtr4FieldName<CfgRpf<std::string>>(field_name)->Mutable(index) = new_val;
  return old_val;
}

}  // namespace oneflow
