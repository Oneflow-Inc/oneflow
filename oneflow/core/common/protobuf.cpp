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

#define DECLARE_GETTER_FUNC_HEADER(type)                      \
 template <>                                                  \
 type GetValFromPbMessage<type>(const PbMessage& msg,         \
                                const std::string& field_name)

#define DECLARE_SETTER_FUNC_HEADER(type)                                      \
  template <>                                                                 \
  void SetValInPbMessage<type>(PbMessage* msg, const std::string& field_name, \
                               const type &val)                               \

#define DEFINE_MESSAGE_VAL_GETTER_AND_SETTER(message_type)               \
  DECLARE_GETTER_FUNC_HEADER(message_type) {                             \
    PROTOBUF_REFLECTION(msg, field_name);                                \
    return *dynamic_cast<const message_type *>(&r->GetMessage(msg, fd)); \
  }                                                                      \
  DECLARE_SETTER_FUNC_HEADER(message_type) {                             \
    PROTOBUF_REFLECTION((*msg), field_name);                             \
    r->MutableMessage(msg, fd)->CopyFrom(val);                           \
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

#define DEFINE_VECTOR_VAL_GETTER_AND_SETTER(vec_type, vec_type_name) \
  DECLARE_GETTER_FUNC_HEADER(vec_type) {                             \
    PROTOBUF_REFLECTION(msg, field_name);                            \
    int32_t field_size = r->FieldSize(msg, fd);                      \
    vec_type retval(field_size);                                     \
    for (int i = 0; i < field_size; ++i) {                           \
      retval[i] = r->Get##vec_type_name(msg, fd, i);                 \
    }                                                                \
    return std::move(retval);                                        \
  }                                                                  \
  DECLARE_SETTER_FUNC_HEADER(vec_type) {                             \
    PROTOBUF_REFLECTION((*msg), field_name);                         \
    for (int i = 0; i < val.size(); ++i) {                           \
      r->Set##vec_type_name(msg, fd, i, val[i]);                     \
    }                                                                \
  }

#define MAKE_REPEATED_TUPLE_SEQ(type, type_name)  \
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
 
OF_PP_FOR_EACH_TUPLE(DEFINE_VECTOR_VAL_GETTER_AND_SETTER,
                     PROTOBUF_BASIC_REPEATED_DATA_TYPE_SEQ);

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
