#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

// txt file
bool TryParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  std::ifstream in_stream(file_path.c_str(), std::ifstream::in);
  google::protobuf::io::IstreamInputStream input(&in_stream);
  return google::protobuf::TextFormat::Parse(&input, proto);
}
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  CHECK(TryParseProtoFromTextFile(file_path, proto));
}
void PrintProtoToTextFile(const PbMessage& proto, const std::string& file_path) {
  std::ofstream out_stream(file_path.c_str(), std::ofstream::out | std::ofstream::trunc);
  google::protobuf::io::OstreamOutputStream output(&out_stream);
  CHECK(google::protobuf::TextFormat::Print(proto, &output));
}

bool HasFieldInPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_GET_FIELDDESC(msg, field_name);
  return fd != nullptr;
}

const PbFd* GetPbFdFromPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_GET_FIELDDESC(msg, field_name);
  CHECK_NOTNULL(fd);
  return fd;
}

#define DEFINE_GET_VAL_FROM_PBMESSAGE(cpp_type, pb_type_name)                                   \
  template<>                                                                                    \
  cpp_type GetValFromPbMessage<cpp_type>(const PbMessage& msg, const std::string& field_name) { \
    PROTOBUF_REFLECTION(msg, field_name);                                                       \
    return r->Get##pb_type_name(msg, fd);                                                       \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_GET_VAL_FROM_PBMESSAGE,
                     PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(const PbMessage&, Message))

int32_t GetEnumFromPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetEnumValue(msg, fd);
}

#define DEFINE_SET_VAL_IN_PBMESSAGE(cpp_type, pb_type_name)                                    \
  template<>                                                                                   \
  void SetValInPbMessage(PbMessage* msg, const std::string& field_name, const cpp_type& val) { \
    PROTOBUF_REFLECTION((*msg), field_name);                                                   \
    r->Set##pb_type_name(msg, fd, val);                                                        \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_SET_VAL_IN_PBMESSAGE, PROTOBUF_BASIC_DATA_TYPE_SEQ)

PbMessage* MutableMessageInPbMessage(PbMessage* msg, const std::string& field_name) {
  PROTOBUF_REFLECTION((*msg), field_name);
  return r->MutableMessage(msg, fd);
}

PbMessage* MutableRepeatedMessageInPbMessage(PbMessage* msg, const std::string& field_name,
                                             int index) {
  PROTOBUF_REFLECTION((*msg), field_name);
  return r->MutableRepeatedMessage(msg, fd, index);
}

const PbMessage& GetMessageInPbMessage(const PbMessage& msg, int field_index) {
  const auto* d = const_cast<google::protobuf::Descriptor*>(msg.GetDescriptor());
  const auto* fd = const_cast<PbFd*>(d->FindFieldByNumber(field_index));
  CHECK_NOTNULL(fd);
  const auto* r = const_cast<google::protobuf::Reflection*>(msg.GetReflection());
  return r->GetMessage(msg, fd);
}

PbMessage* MutableMessageInPbMessage(PbMessage* msg, int field_index) {
  const auto* d = const_cast<google::protobuf::Descriptor*>(msg->GetDescriptor());
  const auto* fd = const_cast<PbFd*>(d->FindFieldByNumber(field_index));
  CHECK_NOTNULL(fd);
  const auto* r = const_cast<google::protobuf::Reflection*>(msg->GetReflection());
  return r->MutableMessage(msg, fd);
}

#define DEFINE_ADD_VAL_IN_PBRF(cpp_type, pb_type_name)                                    \
  template<>                                                                              \
  void AddValInPbRf(PbMessage* msg, const std::string& field_name, const cpp_type& val) { \
    PROTOBUF_REFLECTION((*msg), field_name);                                              \
    r->Add##pb_type_name(msg, fd, val);                                                   \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ADD_VAL_IN_PBRF, PROTOBUF_BASIC_DATA_TYPE_SEQ)

PersistentOutStream& operator<<(PersistentOutStream& out_stream, const PbMessage& msg) {
  std::string msg_bin;
  msg.SerializeToString(&msg_bin);
  int64_t msg_size = msg_bin.size();
  CHECK_GT(msg_size, 0);
  out_stream << msg_size << msg_bin;
  return out_stream;
}

}  // namespace oneflow
