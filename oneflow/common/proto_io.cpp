#include "common/proto_io.h"
#include <stdint.h>
#include <fcntl.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

namespace oneflow {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Descriptor;
using google::protobuf::Reflection;
using google::protobuf::FieldDescriptor;

// string
void ParseProtoFromString(const std::string& str, PbMessage* proto) {
  CHECK(google::protobuf::TextFormat::ParseFromString(str, proto));
}
void PrintProtoToString(const PbMessage& proto, std::string* str) {
  CHECK(google::protobuf::TextFormat::PrintToString(proto, str));
}

// txt file
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  int fd = open(file_path.c_str(), O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << file_path;
  FileInputStream input(fd);
  CHECK(google::protobuf::TextFormat::Parse(&input, proto));
  close(fd);
}
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path) {
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream output(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, &output));
  close(fd);
}

#define DEFINE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name) \
ret_type Get##func_name##FromPbMessage(const PbMessage& msg, \
                                       const std::string& field_name) { \
  const Descriptor* d = msg.GetDescriptor(); \
  const FieldDescriptor* fd = d->FindFieldByName(field_name); \
  CHECK_NOTNULL(fd); \
  const Reflection* r = msg.GetReflection(); \
  return r->Get##func_name (msg, fd); \
}

DEFINE_GET_VAL_FROM_PBMESSAGE(std::string, String);
DEFINE_GET_VAL_FROM_PBMESSAGE(int32_t, Int32);
DEFINE_GET_VAL_FROM_PBMESSAGE(uint32_t, UInt32);
DEFINE_GET_VAL_FROM_PBMESSAGE(int64_t, Int64);
DEFINE_GET_VAL_FROM_PBMESSAGE(uint64_t, UInt64);

} // namespace oneflow
