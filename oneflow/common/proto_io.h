#ifndef ONEFLOW_PROTO_IO_H_
#define ONEFLOW_PROTO_IO_H_

#ifdef _MSC_VER
#include <io.h>
#endif
#include <string>
#include "common/util.h"
#include "operator/op_conf.pb.h"
#include "operator/operator.pb.h"
#include "google/protobuf/message.h"
#include "google/protobuf/descriptor.h"

namespace oneflow {

using PbMessage = google::protobuf::Message;

// Prototxt <-> String
void ParseProtoFromString(const std::string& str, PbMessage* proto);
void PrintProtoToString(const PbMessage& proto, std::string* str);

// Prototxt <-> File
void ParseProtoFromTextFile(const std::string& file_path,
                            PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path);

//
std::string GetValueFromPbMessage(const PbMessage& msg,
                                  const std::string& key);
 
// RepeatedPtrField <-> Vector 
inline std::vector<std::string> PbRepeatedPtrField2Vec(
    const google::protobuf::RepeatedPtrField<std::string>& rpf) {
  return std::vector<std::string> (rpf.begin(), rpf.end());
}
inline google::protobuf::RepeatedPtrField<std::string> Vec2PbRepeatedPtrField (
    const std::vector<std::string>& vec) {
  using RetType = google::protobuf::RepeatedPtrField<std::string>;
  return RetType(vec.begin(), vec.end());
}

// ProtoMap <-> HashMap
inline HashMap<std::string, std::string> PbMap2HashMap(
    const google::protobuf::Map<std::string, std::string>& pb_map) {
  return HashMap<std::string, std::string> (pb_map.begin(), pb_map.end());
}
inline google::protobuf::Map<std::string, std::string> HashMap2PbMap( 
    const HashMap<std::string, std::string>& hash_map) {
  using RetType = google::protobuf::Map<std::string, std::string>;
  return RetType(hash_map.begin(), hash_map.end());
}

} // namespace caffe

#endif // ONEFLOW_PROTO_IO_H_
