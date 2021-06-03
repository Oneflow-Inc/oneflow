/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

// parse protobuf message from .prototxt file
bool TryParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  std::ifstream in_stream(file_path.c_str(), std::ifstream::in);
  google::protobuf::io::IstreamInputStream input(&in_stream);
  return google::protobuf::TextFormat::Parse(&input, proto);
}

void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  CHECK(TryParseProtoFromTextFile(file_path, proto));
}

// parse protobuf message from .pb file
bool TryParseProtoFromPbFile(const std::string& file_path, PbMessage* proto) {
  std::ifstream in_stream(file_path.c_str(), std::ifstream::in | std::ifstream::binary);
  return proto->ParseFromIstream(&in_stream);
}

void ParseProtoFromPbFile(const std::string& file_path, PbMessage* proto) {
  CHECK(TryParseProtoFromPbFile(file_path, proto));
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

bool FieldDefinedInPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_GET_FIELDDESC(msg, field_name);
  return fd != nullptr;
}

#define DEFINE_GET_VAL_FROM_PBMESSAGE(cpp_type, pb_type_name)                                   \
  template<>                                                                                    \
  cpp_type GetValFromPbMessage<cpp_type>(const PbMessage& msg, const std::string& field_name) { \
    PROTOBUF_REFLECTION(msg, field_name);                                                       \
    return r->Get##pb_type_name(msg, fd);                                                       \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_GET_VAL_FROM_PBMESSAGE,
                     PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(const PbMessage&, Message))

#define DEFINE_SET_VAL_IN_PBMESSAGE(cpp_type, pb_type_name)                                    \
  template<>                                                                                   \
  void SetValInPbMessage(PbMessage* msg, const std::string& field_name, const cpp_type& val) { \
    PROTOBUF_REFLECTION((*msg), field_name);                                                   \
    r->Set##pb_type_name(msg, fd, val);                                                        \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_SET_VAL_IN_PBMESSAGE, PROTOBUF_BASIC_DATA_TYPE_SEQ)

const PbMessage& GetMessageInPbMessage(const PbMessage& msg, const std::string& field_name) {
  PROTOBUF_REFLECTION(msg, field_name);
  return r->GetMessage(msg, fd);
}

PbMessage* MutableMessageInPbMessage(PbMessage* msg, const std::string& field_name) {
  PROTOBUF_REFLECTION((*msg), field_name);
  return r->MutableMessage(msg, fd);
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

#define DECLARE_GETTER_FUNC_HEADER(type) \
  template<>                             \
  type GetValFromPbMessage<type>(const PbMessage& msg, const std::string& field_name)

#define DECLARE_SETTER_FUNC_HEADER(type) \
  template<>                             \
  void SetValInPbMessage<type>(PbMessage * msg, const std::string& field_name, const type& val)

#define DEFINE_MESSAGE_VAL_GETTER_AND_SETTER(message_type)              \
  DECLARE_GETTER_FUNC_HEADER(message_type) {                            \
    PROTOBUF_REFLECTION(msg, field_name);                               \
    return *dynamic_cast<const message_type*>(&r->GetMessage(msg, fd)); \
  }                                                                     \
  DECLARE_SETTER_FUNC_HEADER(message_type) {                            \
    PROTOBUF_REFLECTION((*msg), field_name);                            \
    r->MutableMessage(msg, fd)->CopyFrom(val);                          \
  }

DEFINE_MESSAGE_VAL_GETTER_AND_SETTER(ShapeProto);

#define DEFINE_ENUM_VAL_GETTER_AND_SETTER(enum_type)         \
  DECLARE_GETTER_FUNC_HEADER(enum_type) {                    \
    PROTOBUF_REFLECTION(msg, field_name);                    \
    return static_cast<enum_type>(r->GetEnumValue(msg, fd)); \
  }                                                          \
  DECLARE_SETTER_FUNC_HEADER(enum_type) {                    \
    PROTOBUF_REFLECTION((*msg), field_name);                 \
    r->SetEnumValue(msg, fd, val);                           \
  }

DEFINE_ENUM_VAL_GETTER_AND_SETTER(DataType);

#define DEFINE_VECTOR_VAL_GETTER_AND_SETTER(vec_type, vec_type_name)                        \
  DECLARE_GETTER_FUNC_HEADER(vec_type) {                                                    \
    PROTOBUF_REFLECTION(msg, field_name);                                                   \
    int32_t field_size = r->FieldSize(msg, fd);                                             \
    vec_type retval(field_size);                                                            \
    for (int i = 0; i < field_size; ++i) { retval[i] = r->Get##vec_type_name(msg, fd, i); } \
    return retval;                                                                          \
  }                                                                                         \
  DECLARE_SETTER_FUNC_HEADER(vec_type) {                                                    \
    PROTOBUF_REFLECTION((*msg), field_name);                                                \
    for (int i = 0; i < val.size(); ++i) { r->Set##vec_type_name(msg, fd, i, val[i]); }     \
  }

#define MAKE_REPEATED_TUPLE_SEQ(type, type_name) \
  OF_PP_MAKE_TUPLE_SEQ(std::vector<type>, Repeated##type_name)

#define PROTOBUF_BASIC_REPEATED_DATA_TYPE_SEQ  \
  MAKE_REPEATED_TUPLE_SEQ(std::string, String) \
  MAKE_REPEATED_TUPLE_SEQ(int32_t, Int32)      \
  MAKE_REPEATED_TUPLE_SEQ(uint32_t, UInt32)    \
  MAKE_REPEATED_TUPLE_SEQ(int64_t, Int64)      \
  MAKE_REPEATED_TUPLE_SEQ(uint64_t, UInt64)    \
  MAKE_REPEATED_TUPLE_SEQ(float, Float)        \
  MAKE_REPEATED_TUPLE_SEQ(double, Double)      \
  MAKE_REPEATED_TUPLE_SEQ(int16_t, EnumValue)  \
  MAKE_REPEATED_TUPLE_SEQ(bool, Bool)

OF_PP_FOR_EACH_TUPLE(DEFINE_VECTOR_VAL_GETTER_AND_SETTER, PROTOBUF_BASIC_REPEATED_DATA_TYPE_SEQ);

#define DEFINE_ADD_VAL_IN_PBRF(cpp_type, pb_type_name)                                    \
  template<>                                                                              \
  void AddValInPbRf(PbMessage* msg, const std::string& field_name, const cpp_type& val) { \
    PROTOBUF_REFLECTION((*msg), field_name);                                              \
    r->Add##pb_type_name(msg, fd, val);                                                   \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ADD_VAL_IN_PBRF, PROTOBUF_BASIC_DATA_TYPE_SEQ)

std::pair<std::string, int32_t> GetFieldNameAndIndex4StrVal(const std::string& fd_name_with_idx) {
  std::string field_name;
  int32_t idx = 0;
  CHECK_GE(idx, 0);
  GetPrefixAndIndex(fd_name_with_idx, &field_name, &idx);
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
  return TryGetPrefixAndIndex(fd_name_may_have_idx, &field_name, &index);
}

std::string ReplaceStrValInPbFdOrPbRpf(PbMessage* msg, const std::string& fd_name_may_have_idx,
                                       const std::string& new_val) {
  const PbFd* fd = msg->GetDescriptor()->FindFieldByName(fd_name_may_have_idx);
  std::string old_val;
  if (fd) {
    old_val = GetValFromPbMessage<std::string>(*msg, fd_name_may_have_idx);
    SetValInPbMessage<std::string>(msg, fd_name_may_have_idx, new_val);
  } else {
    const std::pair<std::string, int32_t> prefix_idx =
        GetFieldNameAndIndex4StrVal(fd_name_may_have_idx);
    old_val = GetPbRpfFromPbMessage<std::string>(*msg, prefix_idx.first).Get(prefix_idx.second);
    PbRpf<std::string>* rpf = MutPbRpfFromPbMessage<std::string>(msg, prefix_idx.first);
    *rpf->Mutable(prefix_idx.second) = new_val;
  }
  return old_val;
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
