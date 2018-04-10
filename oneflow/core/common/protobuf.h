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
#include "oneflow/core/persistence/persistent_out_stream.h"

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
  OF_PP_MAKE_TUPLE_SEQ(bool, Bool)

#define PROTOBUF_GET_FIELDDESC(msg, field_name)                            \
  auto d = const_cast<google::protobuf::Descriptor*>(msg.GetDescriptor()); \
  auto fd = const_cast<google::protobuf::FieldDescriptor*>(                \
      d->FindFieldByName(field_name));

#define PROTOBUF_REFLECTION(msg, field_name) \
  PROTOBUF_GET_FIELDDESC(msg, field_name)    \
  CHECK_NOTNULL(fd);                         \
  auto r = const_cast<google::protobuf::Reflection*>(msg.GetReflection());

// Prototxt <-> File
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto, const std::string& file_path);

// Does PbMessage have the field_name
bool HasFieldInPbMessage(const PbMessage&, const std::string& field_name);

// Get From PbMessage

template<typename T>
T GetValFromPbMessage(const PbMessage&, const std::string& field_name);

int32_t GetEnumFromPbMessage(const PbMessage&, const std::string& field_name);

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

template<typename T>
void SetValInPbMessage(PbMessage* msg, const std::string& field_name,
                       const T& val);

PbMessage* MutableMessageInPbMessage(PbMessage*, const std::string& field_name);

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

template<typename T = PbMessage>
const T* GetMsgPtrFromPbMessage(const PbMessage& msg,
                                const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  if (r->HasField(msg, fd)) {
    return static_cast<const T*>(
        &(GetValFromPbMessage<const PbMessage&>(msg, field_name)));
  } else {
    return nullptr;
  }
}

// Persistent

PersistentOutStream& operator<<(PersistentOutStream&, const PbMessage&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PROTOBUF_H_
