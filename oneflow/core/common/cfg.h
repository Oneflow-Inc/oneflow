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
#ifndef ONEFLOW_CORE_COMMON_CFG_H_
#define ONEFLOW_CORE_COMMON_CFG_H_

#include <vector>
#include <unordered_map>
#include <glog/logging.h>
#include "oneflow/cfg/message.h"
#include "oneflow/cfg/map_field.h"
#include "oneflow/cfg/repeated_field.h"

namespace oneflow {

using CfgMessage = ::oneflow::cfg::Message;
template<typename T>
using CfgRf = ::oneflow::cfg::_RepeatedField_<T>;
template<typename T>
using CfgRpf = ::oneflow::cfg::_RepeatedField_<T>;
template<typename T1, typename T2>
using CfgMapPair = std::pair<T1, T2>;
template<typename K, typename V>
using CfgMap = ::oneflow::cfg::_MapField_<K, V>;

// Does CfgMessage have the field_name
bool FieldDefinedInCfgMessage(const CfgMessage& msg, const std::string& field_name);

// Get From CfgMessage
template<typename T>
T GetValFromCfgMessage(const CfgMessage& msg, const std::string& field_name) {
  return *CHECK_NOTNULL(msg.FieldPtr4FieldName<T>(field_name));
}

template<typename T>
const CfgRf<T>& GetCfgRfFromCfgMessage(const CfgMessage& msg, const std::string& field_name) {
  return *CHECK_NOTNULL(msg.FieldPtr4FieldName<CfgRf<T>>(field_name));
}

template<typename T>
const CfgRpf<T>& GetCfgRpfFromCfgMessage(const CfgMessage& msg, const std::string& field_name) {
  return *CHECK_NOTNULL(msg.FieldPtr4FieldName<CfgRpf<T>>(field_name));
}

template<typename T>
CfgRpf<T>* MutCfgRpfFromCfgMessage(CfgMessage* msg, const std::string& field_name) {
  return CHECK_NOTNULL(msg->MutableFieldPtr4FieldName<CfgRpf<T>>(field_name));
}

// Set In CfgMessage

template<typename T>
void SetValInCfgMessage(CfgMessage* msg, const std::string& field_name, const T& val) {
  *CHECK_NOTNULL(msg->MutableFieldPtr4FieldName<T>(field_name)) = val;
}

const CfgMessage& GetMessageInCfgMessage(const CfgMessage& msg, int field_index);
const CfgMessage& GetMessageInCfgMessage(const CfgMessage& msg, const std::string& field_name);

CfgMessage* MutableMessageInCfgMessage(CfgMessage*, const std::string& field_name);
CfgMessage* MutableMessageInCfgMessage(CfgMessage*, int field_index);

// Get/Replace str val maybe repeated;  field_name with index is like "name_0"
std::string GetStrValInCfgFdOrCfgRpf(const CfgMessage& msg,
                                     const std::string& fd_name_may_have_idx);
bool HasStrFieldInCfgFdOrCfgRpf(const CfgMessage& msg, const std::string& fd_name_may_have_idx);
// return old value
std::string ReplaceStrValInCfgFdOrCfgRpf(CfgMessage* msg, const std::string& fd_name_may_have_idx,
                                         const std::string& new_val);

// Add In CfgMessage RepeatedField

template<typename T>
void AddValInCfgRf(CfgMessage* msg, const std::string& field_name, const T& val) {
  CfgRf<T>* repeated_field = msg->MutableFieldPtr4FieldName<CfgRf<T>>(field_name);
  CHECK_NOTNULL(repeated_field)->Add(val);
}

// CfgRf <-> std::vector

template<typename T>
inline std::vector<T> CfgRf2StdVec(const CfgRf<T>& rf) {
  return std::vector<T>(rf.begin(), rf.end());
}

template<typename T>
inline CfgRf<T> StdVec2CfgRf(const std::vector<T>& vec) {
  return CfgRf<T>(vec.begin(), vec.end());
}

// CfgRpf <-> std::vector
template<typename T>
inline std::vector<T> CfgRpf2StdVec(const CfgRpf<T>& rpf) {
  return std::vector<T>(rpf.begin(), rpf.end());
}

template<typename T>
inline CfgRpf<T> StdVec2CfgRpf(const std::vector<T>& vec) {
  return CfgRpf<T>(vec.begin(), vec.end());
}

// CfgMap <-> std::unordered_map
template<typename K, typename V>
std::unordered_map<K, V> CfgMap2HashMap(const CfgMap<K, V>& cfg_map) {
  return std::unordered_map<K, V>(cfg_map.begin(), cfg_map.end());
}

template<typename K, typename V>
CfgMap<K, V> HashMap2CfgMap(const std::unordered_map<K, V>& hash_map) {
  return CfgMap<K, V>(hash_map.begin(), hash_map.end());
}

// If value exists in RepeatedField
template<typename T>
bool IsInRepeatedField(const CfgRf<T>& repeated_field, const T& value) {
  return std::find(repeated_field.begin(), repeated_field.end(), value) != repeated_field.end();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CFG_H_
