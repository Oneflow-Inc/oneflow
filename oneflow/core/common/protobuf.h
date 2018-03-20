#ifndef ONEFLOW_CORE_COMMON_PROTOBUF_H_
#define ONEFLOW_CORE_COMMON_PROTOBUF_H_

#ifdef _MSC_VER
#include <io.h>
#endif
#include <google/protobuf/descriptor.h>
#include <google/protobuf/map.h>
#include <google/protobuf/message.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

using PbMessage = google::protobuf::Message;
template<typename T>
using PbRf = google::protobuf::RepeatedField<T>;
template<typename T>
using PbRpf = google::protobuf::RepeatedPtrField<T>;
template<typename T1, typename T2>
using PbMapPair = google::protobuf::MapPair<T1, T2>;

#define PROTOBUF_BASIC_DATA_TYPE_SEQ        \
  OF_PP_MAKE_TUPLE_SEQ(std::string, String) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, Int32)      \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, UInt32)    \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, Int64)      \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, UInt64)    \
  OF_PP_MAKE_TUPLE_SEQ(float, Float)        \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, EnumValue)  \
  OF_PP_MAKE_TUPLE_SEQ(bool, Bool)

#define PROTOBUF_REFLECTION(msg, field_name)                               \
  auto d = const_cast<google::protobuf::Descriptor*>(msg.GetDescriptor()); \
  auto fd = const_cast<google::protobuf::FieldDescriptor*>(                \
      d->FindFieldByName(field_name));                                     \
  CHECK_NOTNULL(fd);                                                       \
  auto r = const_cast<google::protobuf::Reflection*>(msg.GetReflection());

// Prototxt <-> File
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto, const std::string& file_path);

// Get From PbMessage

#define DECLARE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name)    \
  ret_type Get##func_name##FromPbMessage(const PbMessage& msg, \
                                         const std::string& field_name);

OF_PP_FOR_EACH_TUPLE(DECLARE_GET_VAL_FROM_PBMESSAGE,
                     PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(
                         const PbMessage&, Message))

#undef DECLARE_GET_VAL_FROM_PBMESSAGE

template<typename T>
const PbRf<T>& GetPbRfFromPbMessage(const PbMessage& msg,
                                    const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetRepeatedField<T>(msg, fd);
}

template<typename T>
const PbRpf<T>& GetPbRpfFromPbMessage(const PbMessage& msg,
                                      const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetRepeatedPtrField<T>(msg, fd);
}

// Set In PbMessage

#define DECLARE_SET_VAL_IN_PBMESSAGE(val_type, func_name) \
  void Set##func_name##InPbMessage(                       \
      PbMessage* msg, const std::string& field_name, val_type val);

OF_PP_FOR_EACH_TUPLE(DECLARE_SET_VAL_IN_PBMESSAGE, PROTOBUF_BASIC_DATA_TYPE_SEQ)

PbMessage* MutableMessageInPbMessage(PbMessage*, const std::string& field_name);

#undef DECLARE_SET_VAL_IN_PBMESSAGE

// Add In PbMessage RepeatedField

#define DECLARE_ADD_VAL_IN_PBRF(val_type, func_name)                         \
  void Add##func_name##InPbRf(PbMessage* msg, const std::string& field_name, \
                              val_type val);

OF_PP_FOR_EACH_TUPLE(DECLARE_ADD_VAL_IN_PBRF, PROTOBUF_BASIC_DATA_TYPE_SEQ)

#undef DECLARE_ADD_VAL_IN_PBRF

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
