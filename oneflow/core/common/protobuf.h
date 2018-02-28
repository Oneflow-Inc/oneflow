#ifndef ONEFLOW_CORE_COMMON_PROTOBUF_H_
#define ONEFLOW_CORE_COMMON_PROTOBUF_H_

#ifdef _MSC_VER
#include <io.h>
#endif
#include <google/protobuf/descriptor.h>
#include <google/protobuf/map.h>
#include <google/protobuf/message.h>
#include "oneflow/core/common/util.h"

namespace oneflow {

using PbMessage = google::protobuf::Message;
template<typename T>
using PbRf = google::protobuf::RepeatedField<T>;
template<typename T>
using PbRpf = google::protobuf::RepeatedPtrField<T>;
template<typename T1, typename T2>
using PbMapPair = google::protobuf::MapPair<T1, T2>;

// Prototxt <-> File
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto, const std::string& file_path);

// Get From PbMessage

#define DECLARE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name)    \
  ret_type Get##func_name##FromPbMessage(const PbMessage& msg, \
                                         const std::string& field_name);

DECLARE_GET_VAL_FROM_PBMESSAGE(std::string, String);
DECLARE_GET_VAL_FROM_PBMESSAGE(int32_t, Int32);
DECLARE_GET_VAL_FROM_PBMESSAGE(uint32_t, UInt32);
DECLARE_GET_VAL_FROM_PBMESSAGE(int64_t, Int64);
DECLARE_GET_VAL_FROM_PBMESSAGE(uint64_t, UInt64);
DECLARE_GET_VAL_FROM_PBMESSAGE(bool, Bool);
DECLARE_GET_VAL_FROM_PBMESSAGE(const PbMessage&, Message);

#undef DECLARE_GET_VAL_FROM_PBMESSAGE

template<typename T>
const PbRf<T>& GetPbRfFromPbMessage(const PbMessage& msg,
                                    const std::string& field_name) {
  const google::protobuf::Descriptor* d = msg.GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = d->FindFieldByName(field_name);
  CHECK_NOTNULL(fd);
  const google::protobuf::Reflection* r = msg.GetReflection();
  return r->GetRepeatedField<T>(msg, fd);
}

// Set In PbMessage

#define DECLARE_SET_VAL_IN_PBMESSAGE(val_type, func_name) \
  void Set##func_name##InPbMessage(                       \
      PbMessage* msg, const std::string& field_name, val_type val);

DECLARE_SET_VAL_IN_PBMESSAGE(std::string, String);
DECLARE_SET_VAL_IN_PBMESSAGE(int32_t, Int32);
DECLARE_SET_VAL_IN_PBMESSAGE(uint32_t, UInt32);
DECLARE_SET_VAL_IN_PBMESSAGE(int64_t, Int64);
DECLARE_SET_VAL_IN_PBMESSAGE(uint64_t, UInt64);
DECLARE_SET_VAL_IN_PBMESSAGE(bool, Bool);

#undef DECLARE_SET_VAL_IN_PBMESSAGE

// Add In PbMessage RepeatedField

#define DECLARE_ADD_VAL_IN_PBRF(val_type, func_name)                         \
  void Add##func_name##InPbRf(PbMessage* msg, const std::string& field_name, \
                              val_type val);

DECLARE_ADD_VAL_IN_PBRF(std::string, String);
DECLARE_ADD_VAL_IN_PBRF(int32_t, Int32);
DECLARE_ADD_VAL_IN_PBRF(uint32_t, UInt32);
DECLARE_ADD_VAL_IN_PBRF(int64_t, Int64);
DECLARE_ADD_VAL_IN_PBRF(uint64_t, UInt64);
DECLARE_ADD_VAL_IN_PBRF(bool, Bool);

#undef DECLARE_ADD_VAL_IN_PBRF

// Alias PbType

#define ALIAS_PB_TYPE(type, name) using Pb##name = google::protobuf::type;

ALIAS_PB_TYPE(int32, Int32);
ALIAS_PB_TYPE(int64, Int64);
ALIAS_PB_TYPE(uint32, UInt32);
ALIAS_PB_TYPE(uint64, UInt64);

#undef ALIAS_PB_TYPE

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

// operator
inline bool operator==(const google::protobuf::MessageLite& lhs,
                       const google::protobuf::MessageLite& rhs) {
  return lhs.SerializeAsString() == rhs.SerializeAsString();
}

inline bool operator!=(const google::protobuf::MessageLite& lhs,
                       const google::protobuf::MessageLite& rhs) {
  return !(lhs == rhs);
}

// Hack Oneof Getter

#define OF_PB_POINTER_GET(obj, field) \
  obj.has_##field() ? &(obj.field()) : nullptr

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PROTOBUF_H_
