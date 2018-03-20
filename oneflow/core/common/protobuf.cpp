#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace oneflow {

// txt file
void ParseProtoFromTextFile(const std::string& file_path, PbMessage* proto) {
  std::ifstream in_stream(file_path.c_str(), std::ifstream::in);
  google::protobuf::io::IstreamInputStream input(&in_stream);
  CHECK(google::protobuf::TextFormat::Parse(&input, proto));
}
void PrintProtoToTextFile(const PbMessage& proto,
                          const std::string& file_path) {
  std::ofstream out_stream(file_path.c_str(),
                           std::ofstream::out | std::ofstream::trunc);
  google::protobuf::io::OstreamOutputStream output(&out_stream);
  CHECK(google::protobuf::TextFormat::Print(proto, &output));
}

#define DEFINE_GET_VAL_FROM_PBMESSAGE(ret_type, func_name)                \
  ret_type Get##func_name##FromPbMessage(const PbMessage& msg,            \
                                         const std::string& field_name) { \
    PROTOBUF_REFLECTION(msg, field_name);                                 \
    return r->Get##func_name(msg, fd);                                    \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_GET_VAL_FROM_PBMESSAGE,
                     PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(
                         const PbMessage&, Message))

#define DEFINE_SET_VAL_IN_PBMESSAGE(val_type, func_name)             \
  void Set##func_name##InPbMessage(                                  \
      PbMessage* msg, const std::string& field_name, val_type val) { \
    PROTOBUF_REFLECTION((*msg), field_name);                         \
    r->Set##func_name(msg, fd, val);                                 \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_SET_VAL_IN_PBMESSAGE, PROTOBUF_BASIC_DATA_TYPE_SEQ)

PbMessage* MutableMessageInPbMessage(PbMessage* msg,
                                     const std::string& field_name) {
  PROTOBUF_REFLECTION((*msg), field_name);
  return r->MutableMessage(msg, fd);
}

#define DEFINE_ADD_VAL_IN_PBRF(val_type, func_name)                          \
  void Add##func_name##InPbRf(PbMessage* msg, const std::string& field_name, \
                              val_type val) {                                \
    PROTOBUF_REFLECTION((*msg), field_name);                                 \
    r->Add##func_name(msg, fd, val);                                         \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_ADD_VAL_IN_PBRF, PROTOBUF_BASIC_DATA_TYPE_SEQ)

}  // namespace oneflow
