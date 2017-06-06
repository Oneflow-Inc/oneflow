#ifndef ONEFLOW_CORE_COMMON_PROTOBUF_H_
#define ONEFLOW_CORE_COMMON_PROTOBUF_H_

#ifdef _MSC_VER
#include <io.h>
#endif
#include <string>
#include "oneflow/core/common/util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.pb.h"
#include "google/protobuf/message.h"
#include "google/protobuf/descriptor.h"

namespace oneflow {

using PbMessage = google::protobuf::Message;
template<typename T>
using PbRf = google::protobuf::RepeatedField<T>;
template<typename T>
using PbRpf = google::protobuf::RepeatedPtrField<T>;
template<typename T1, typename T2>
using PbMapPair = google::protobuf::MapPair<T1, T2>;

// Prototxt <-> String
void ParseProtoFromString(const std::string& str, PbMessage* proto);
void PrintProtoToString(const PbMessage& proto, std::string* str);

// Prototxt <-> File
void ParseProtoFromTextFile(const std::string& file_path,
                            PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path);

// Get From PbMessage

#define DECLARE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name) \
ret_type Get##func_name##FromPbMessage(const PbMessage& msg, \
                                       const std::string& field_name);

DECLARE_GET_VAL_FROM_PBMESSAGE(std::string, String);
DECLARE_GET_VAL_FROM_PBMESSAGE(int32_t, Int32);
DECLARE_GET_VAL_FROM_PBMESSAGE(uint32_t, UInt32);
DECLARE_GET_VAL_FROM_PBMESSAGE(int64_t, Int64);
DECLARE_GET_VAL_FROM_PBMESSAGE(uint64_t, UInt64);

#undef DECLARE_GET_VAL_FROM_PBMESSAGE

// Alias PbType

#define ALIAS_PB_TYPE(type, name) \
using Pb##name = google::protobuf::type; \

ALIAS_PB_TYPE(int32, Int32);
ALIAS_PB_TYPE(int64, Int64);
ALIAS_PB_TYPE(uint32, UInt32);
ALIAS_PB_TYPE(uint64, UInt64);

#undef ALIAS_PB_TYPE

// PbRpf <-> std::vector 
inline std::vector<std::string> PbVec2StdVec(
    const PbRpf<std::string>& rpf) {
  return std::vector<std::string> (rpf.begin(), rpf.end());
}
inline PbRpf<std::string> StdVec2PbVec (
    const std::vector<std::string>& vec) {
  using RetType = PbRpf<std::string>;
  return RetType(vec.begin(), vec.end());
}

// ProtoMap <-> HashMap
template<typename K, typename V>
HashMap<K, V> PbMap2HashMap(const google::protobuf::Map<K, V>& pb_map) {
  return HashMap<K, V> (pb_map.begin(), pb_map.end());
}

template<typename K, typename V>
google::protobuf::Map<K, V> HashMap2PbMap(const HashMap<K, V>& hash_map) {
  using RetType = google::protobuf::Map<std::string, std::string>;
  return RetType(hash_map.begin(), hash_map.end());
}

} // namespace caffe

#endif // ONEFLOW_CORE_COMMON_PROTOBUF_H_
