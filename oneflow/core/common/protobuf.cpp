#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

namespace {

bool TryGetFieldNameAndIndex4StrVal(const std::string& fd_name_with_idx, std::string* field_name,
                                    int32_t* index) {
  const size_t underline_pos = fd_name_with_idx.rfind('_');
  if (underline_pos == std::string::npos) { return false; }
  if (underline_pos == 0) { return false; }
  if (underline_pos == fd_name_with_idx.size() - 1) { return false; }
  *field_name = fd_name_with_idx.substr(0, underline_pos);
  std::string index_str = fd_name_with_idx.substr(underline_pos + 1);
  if (IsStrInt(index_str) == false) { return false; }
  *index = oneflow_cast<int32_t>(index_str);
  return *index >= 0;
}

}  // namespace

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

std::string PbMessage2TxtString(const PbMessage& proto) {
  std::string str;
  PbMessage2TxtString(proto, &str);
  return str;
}

void PbMessage2TxtString(const PbMessage& proto, std::string* str) {
  google::protobuf::TextFormat::PrintToString(proto, str);
}

bool TxtString2PbMessage(const std::string& proto_str, PbMessage* msg) {
  return google::protobuf::TextFormat::ParseFromString(proto_str, msg);
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

std::pair<std::string, int32_t> GetFieldNameAndIndex4StrVal(const std::string& fd_name_with_idx) {
  const size_t underline_pos = fd_name_with_idx.rfind('_');
  CHECK_NE(underline_pos, std::string::npos);
  CHECK_GT(underline_pos, 0);
  CHECK_LT(underline_pos, fd_name_with_idx.size() - 1);
  const std::string field_name = fd_name_with_idx.substr(0, underline_pos);
  const int32_t idx = oneflow_cast<int32_t>(fd_name_with_idx.substr(underline_pos + 1));
  CHECK_GE(idx, 0);
  return std::make_pair(field_name, idx);
}

std::string GetStrValInPbFdOrPbRpf(const PbMessage& msg, const std::string& fd_name_may_have_idx) {
  const PbFd* fd = msg.GetDescriptor()->FindFieldByName(fd_name_may_have_idx);
  if (fd) {
    return GetValFromPbMessage<std::string>(msg, fd_name_may_have_idx);
  } else {
    const std::pair<std::string, int32_t> prefix_idx =
        GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    return GetPbRpfFromPbMessage<std::string>(msg, prefix_idx.first).Get(prefix_idx.second);
  }
}

bool HasStrFieldInPbFdOrPbRpf(const PbMessage& msg, const std::string& fd_name_may_have_idx) {
  const PbFd* fd = msg.GetDescriptor()->FindFieldByName(fd_name_may_have_idx);
  if (fd != nullptr) { return true; }
  std::string field_name;
  int32_t index = 0;
  return TryGetFieldNameAndIndex4StrVal(fd_name_may_have_idx, &field_name, &index);
}

void ReplaceStrValInPbFdOrPbRpf(PbMessage* msg, const std::string& fd_name_may_have_idx,
                                const std::string& old_val, const std::string& new_val) {
  const PbFd* fd = msg->GetDescriptor()->FindFieldByName(fd_name_may_have_idx);
  if (fd) {
    CHECK_EQ(GetValFromPbMessage<std::string>(*msg, fd_name_may_have_idx), old_val);
    SetValInPbMessage<std::string>(msg, fd_name_may_have_idx, new_val);
  } else {
    const std::pair<std::string, int32_t> prefix_idx =
        GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    CHECK_EQ(GetPbRpfFromPbMessage<std::string>(*msg, prefix_idx.first).Get(prefix_idx.second),
             old_val);
    PbRpf<std::string>* rpf = MutPbRpfFromPbMessage<std::string>(msg, prefix_idx.first);
    *rpf->Mutable(prefix_idx.second) = new_val;
  }
}

PersistentOutStream& operator<<(PersistentOutStream& out_stream, const PbMessage& msg) {
  std::string msg_bin;
  msg.SerializeToString(&msg_bin);
  int64_t msg_size = msg_bin.size();
  CHECK_GT(msg_size, 0);
  out_stream << msg_size << msg_bin;
  return out_stream;
}

bool operator==(const BlobDescProto& lhs, const BlobDescProto& rhs) {
  return PbMd().Equivalent(lhs, rhs);
}

}  // namespace oneflow
