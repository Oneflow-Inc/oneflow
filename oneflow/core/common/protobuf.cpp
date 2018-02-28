#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Reflection;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::OstreamOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

// txt file
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  std::ifstream in_stream(file_path.c_str(), std::ifstream::in);
  IstreamInputStream input(&in_stream);
  CHECK(google::protobuf::TextFormat::Parse(&input, proto));
}
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path) {
  std::ofstream out_stream(file_path.c_str(),
                           std::ofstream::out | std::ofstream::trunc);
  OstreamOutputStream output(&out_stream);
  CHECK(google::protobuf::TextFormat::Print(proto, &output));
}

#define DEFINE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name)                \
  ret_type Get##func_name##FromPbMessage(const PbMessage& msg,            \
                                         const std::string& field_name) { \
    const Descriptor* d = msg.GetDescriptor();                            \
    const FieldDescriptor* fd = d->FindFieldByName(field_name);           \
    CHECK_NOTNULL(fd);                                                    \
    const Reflection* r = msg.GetReflection();                            \
    return r->Get##func_name(msg, fd);                                    \
  }

DEFINE_GET_VAL_FROM_PBMESSAGE(std::string, String);
DEFINE_GET_VAL_FROM_PBMESSAGE(int32_t, Int32);
DEFINE_GET_VAL_FROM_PBMESSAGE(uint32_t, UInt32);
DEFINE_GET_VAL_FROM_PBMESSAGE(int64_t, Int64);
DEFINE_GET_VAL_FROM_PBMESSAGE(uint64_t, UInt64);
DEFINE_GET_VAL_FROM_PBMESSAGE(bool, Bool);
DEFINE_GET_VAL_FROM_PBMESSAGE(const PbMessage&, Message);

#undef DEFINE_GET_VAL_FROM_PBMESSAGE

#define DEFINE_SET_VAL_IN_PBMESSAGE(val_type, func_name)             \
  void Set##func_name##InPbMessage(                                  \
      PbMessage* msg, const std::string& field_name, val_type val) { \
    const Descriptor* d = msg->GetDescriptor();                      \
    const FieldDescriptor* fd = d->FindFieldByName(field_name);      \
    CHECK_NOTNULL(fd);                                               \
    const Reflection* r = msg->GetReflection();                      \
    r->Set##func_name(msg, fd, val);                                 \
  }

DEFINE_SET_VAL_IN_PBMESSAGE(std::string, String);
DEFINE_SET_VAL_IN_PBMESSAGE(int32_t, Int32);
DEFINE_SET_VAL_IN_PBMESSAGE(uint32_t, UInt32);
DEFINE_SET_VAL_IN_PBMESSAGE(int64_t, Int64);
DEFINE_SET_VAL_IN_PBMESSAGE(uint64_t, UInt64);
DEFINE_SET_VAL_IN_PBMESSAGE(bool, Bool);

#undef DEFINE_SET_VAL_IN_PBMESSAGE

#define DEFINE_ADD_VAL_IN_PBRF(val_type, func_name)                          \
  void Add##func_name##InPbRf(PbMessage* msg, const std::string& field_name, \
                              val_type val) {                                \
    const Descriptor* d = msg->GetDescriptor();                              \
    const FieldDescriptor* fd = d->FindFieldByName(field_name);              \
    CHECK_NOTNULL(fd);                                                       \
    const Reflection* r = msg->GetReflection();                              \
    r->Add##func_name(msg, fd, val);                                         \
  }

DEFINE_ADD_VAL_IN_PBRF(std::string, String);
DEFINE_ADD_VAL_IN_PBRF(int32_t, Int32);
DEFINE_ADD_VAL_IN_PBRF(uint32_t, UInt32);
DEFINE_ADD_VAL_IN_PBRF(int64_t, Int64);
DEFINE_ADD_VAL_IN_PBRF(uint64_t, UInt64);
DEFINE_ADD_VAL_IN_PBRF(bool, Bool);

#undef DEFINE_ADD_VAL_IN_PBRF

}  // namespace oneflow
