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
#ifndef ONEFLOW_CORE_COMMON_PROTOBUF_H_
#define ONEFLOW_CORE_COMMON_PROTOBUF_H_

#ifdef _MSC_VER
#include <io.h>
#endif
#include <google/protobuf/descriptor.h>
#include <google/protobuf/map.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/message_differencer.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/register/op_blob_arg.pb.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

using PbMessage = google::protobuf::Message;
template<typename T>
using PbRf = google::protobuf::RepeatedField<T>;
template<typename T>
using PbRpf = google::protobuf::RepeatedPtrField<T>;
template<typename T1, typename T2>
using PbMapPair = google::protobuf::MapPair<T1, T2>;
template<typename K, typename V>
using PbMap = google::protobuf::Map<K, V>;
using PbFd = google::protobuf::FieldDescriptor;
using PbMd = google::protobuf::util::MessageDifferencer;

#define PROTOBUF_BASIC_DATA_TYPE_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ(std::string, String) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, Int32)      \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, UInt32)    \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, Int64)      \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, UInt64)    \
  OF_PP_MAKE_TUPLE_SEQ(float, Float)        \
  OF_PP_MAKE_TUPLE_SEQ(double, Double)      \
  OF_PP_MAKE_TUPLE_SEQ(int16_t, EnumValue)  \
  OF_PP_MAKE_TUPLE_SEQ(bool, Bool)

#define PROTOBUF_GET_FIELDDESC(msg, field_name)                            \
  auto d = const_cast<google::protobuf::Descriptor*>(msg.GetDescriptor()); \
  auto fd = const_cast<PbFd*>(d->FindFieldByName(field_name));

#define PROTOBUF_REFLECTION(msg, field_name) \
  PROTOBUF_GET_FIELDDESC(msg, field_name)    \
  CHECK_NOTNULL(fd);                         \
  auto r = const_cast<google::protobuf::Reflection*>(msg.GetReflection());

// Prototxt <-> File
bool TryParseProtoFromTextFile(const std::string& file_path, PbMessage* proto);
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto);
bool TryParseProtoFromPbFile(const std::string& file_path, PbMessage* proto);
void ParseProtoFromPbFile(const std::string& file_path, PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto, const std::string& file_path);
std::string PbMessage2TxtString(const PbMessage& proto);
void PbMessage2TxtString(const PbMessage& proto, std::string* str);
bool TxtString2PbMessage(const std::string& proto_str, PbMessage* proto);

// Does PbMessage have the field_name
bool FieldDefinedInPbMessage(const PbMessage&, const std::string& field_name);

// Get From PbMessage
template<typename T>
T GetValFromPbMessage(const PbMessage&, const std::string& field_name);

template<typename T>
const PbRf<T>& GetPbRfFromPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetRepeatedField<T>(msg, fd);
}

template<typename T>
const PbRpf<T>& GetPbRpfFromPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetRepeatedPtrField<T>(msg, fd);
}

template<typename T>
PbRpf<T>* MutPbRpfFromPbMessage(PbMessage* msg, const std::string& field_name) {
  PROTOBUF_REFLECTION((*msg), field_name);
  return r->MutableRepeatedPtrField<T>(msg, fd);
}

// Set In PbMessage

template<typename T>
void SetValInPbMessage(PbMessage* msg, const std::string& field_name, const T& val);

const PbMessage& GetMessageInPbMessage(const PbMessage& msg, int field_index);
const PbMessage& GetMessageInPbMessage(const PbMessage& msg, const std::string& field_name);

PbMessage* MutableMessageInPbMessage(PbMessage*, const std::string& field_name);
PbMessage* MutableMessageInPbMessage(PbMessage*, int field_index);

// Get/Replace str val maybe repeated;  field_name with index is like "name_0"
std::pair<std::string, int32_t> GetFieldNameAndIndex4StrVal(const std::string& fd_name_with_idx);
std::string GetStrValInPbFdOrPbRpf(const PbMessage& msg, const std::string& fd_name_may_have_idx);
bool HasStrFieldInPbFdOrPbRpf(const PbMessage& msg, const std::string& fd_name_may_have_idx);
// return old value
std::string ReplaceStrValInPbFdOrPbRpf(PbMessage* msg, const std::string& fd_name_may_have_idx,
                                       const std::string& new_val);

// Add In PbMessage RepeatedField

template<typename T>
void AddValInPbRf(PbMessage*, const std::string& field_name, const T& val);

// PbRf <-> std::vector

template<typename T>
inline std::vector<T> PbRf2StdVec(const PbRf<T>& rf) {
  return std::vector<T>(rf.begin(), rf.end());
}

template<typename T>
inline PbRf<T> StdVec2PbRf(const std::vector<T>& vec) {
  return PbRf<T>(vec.begin(), vec.end());
}

// PbRpf <-> std::vector
template<typename T>
inline std::vector<T> PbRpf2StdVec(const PbRpf<T>& rpf) {
  return std::vector<T>(rpf.begin(), rpf.end());
}

template<typename T>
inline PbRpf<T> StdVec2PbRpf(const std::vector<T>& vec) {
  using RetType = PbRpf<T>;
  return RetType(vec.begin(), vec.end());
}

// ProtoMap <-> HashMap
template<typename K, typename V>
HashMap<K, V> PbMap2HashMap(const google::protobuf::Map<K, V>& pb_map) {
  return HashMap<K, V>(pb_map.begin(), pb_map.end());
}

template<typename K, typename V>
google::protobuf::Map<K, V> HashMap2PbMap(const HashMap<K, V>& hash_map) {
  using RetType = google::protobuf::Map<K, V>;
  return RetType(hash_map.begin(), hash_map.end());
}

// If value exists in RepeatedField
template<typename T>
bool IsInRepeatedField(const PbRf<T>& repeated_field, const T& value) {
  return std::find(repeated_field.cbegin(), repeated_field.cend(), value) != repeated_field.cend();
}

// LBI compare operator

inline bool operator<(const LogicalBlobId& lhs, const LogicalBlobId& rhs) {
  if (lhs.op_name() != rhs.op_name()) { return lhs.op_name() < rhs.op_name(); }
  if (lhs.blob_name() != rhs.blob_name()) { return lhs.blob_name() < rhs.blob_name(); }
  return false;
}

inline bool operator==(const LogicalBlobId& lhs, const LogicalBlobId& rhs) {
  return lhs.op_name() == rhs.op_name() && lhs.blob_name() == rhs.blob_name();
}

inline bool operator!=(const LogicalBlobId& lhs, const LogicalBlobId& rhs) { return !(lhs == rhs); }

inline bool operator==(const OpBlobArg& lhs, const OpBlobArg& rhs) {
  return PbMd().Equals(lhs, rhs);
}

inline bool operator!=(const OpBlobArg& lhs, const OpBlobArg& rhs) { return !(lhs == rhs); }

class BlobDescProto;
bool operator==(const BlobDescProto& lhs, const BlobDescProto& rhs);
inline bool operator!=(const BlobDescProto& lhs, const BlobDescProto& rhs) { return !(lhs == rhs); }

// Persistent

PersistentOutStream& operator<<(PersistentOutStream&, const PbMessage&);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DataType> {
  size_t operator()(const oneflow::DataType data_type) const {
    return std::hash<int64_t>()(data_type);
  }
};

template<>
struct hash<oneflow::LogicalBlobId> {
  size_t operator()(const oneflow::LogicalBlobId& lbi) const {
    const auto& str_hash = std::hash<std::string>();
    return str_hash(lbi.op_name()) ^ str_hash(lbi.blob_name());
  }
};

template<>
struct hash<oneflow::OpBlobArg> {
  size_t operator()(const oneflow::OpBlobArg& oba) const {
    const auto& str_hash = std::hash<std::string>();
    return str_hash(oba.op_name()) ^ str_hash(oba.bn_in_op());
  }
};

template<>
struct hash<oneflow::SbpParallel> {
  size_t operator()(const oneflow::SbpParallel& sbp_parallel) const {
    const auto& str_hash = std::hash<std::string>();
    size_t ret = 0;
    if (sbp_parallel.has_broadcast_parallel()) {
      ret ^= str_hash("B");
    } else if (sbp_parallel.has_partial_sum_parallel()) {
      ret ^= str_hash("P");
    } else if (sbp_parallel.has_split_parallel()) {
      ret ^= str_hash("S");
      ret ^= std::hash<int64_t>()(sbp_parallel.split_parallel().axis());
    } else {
      UNIMPLEMENTED();
    }
    return ret;
  }
};

template<>
struct hash<oneflow::ParallelDistribution> {
  size_t operator()(const oneflow::ParallelDistribution& parallel_distribution) const {
    const auto& sbp_hash = std::hash<oneflow::SbpParallel>();
    size_t hash = 0;
    for (int i = 0; i < parallel_distribution.sbp_parallel_size(); ++i) {
      oneflow::HashCombine(&hash, sbp_hash(parallel_distribution.sbp_parallel(i)));
    }
    return hash;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_PROTOBUF_H_
