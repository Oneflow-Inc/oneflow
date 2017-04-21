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

// string
void ParseProtoFromString(const std::string& str, PbMessage* proto);
void PrintProtoToString(const PbMessage& proto, std::string* str);

// txt file
void ParseProtoFromTextFile(const std::string& file_path,
                            PbMessage* proto);
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path);

std::string GetValueFromPbMessage(const PbMessage& msg,
                                  const std::string& key);
  
void PbRepeatedPtrField2Vec(const google::protobuf::RepeatedPtrField<std::string>& rpf, std::vector<std::string>& vec);

void GPMap2HashMap(const google::protobuf::Map<std::string, std::string>& gmap, HashMap<std::string, std::string>& map );
  
google::protobuf::RepeatedPtrField<std::string> Vec2PbRepeatedPtrField(const std::vector<std::string>& vec);

google::protobuf::Map<std::string, std::string> HashMap2GPMap(HashMap<std::string, std::string>& map);
} // namespace caffe

#endif // ONEFLOW_PROTO_IO_H_
