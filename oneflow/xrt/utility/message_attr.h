#ifndef ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_
#define ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_

#include "glog/logging.h"

#include "oneflow/core/common/protobuf.h"

namespace oneflow {
namespace xrt {
namespace util {

template <typename T>
inline void GetAttr(const PbMessage &message, const std::string &attr_name,
                    T *value) {
  CHECK(HasFieldInPbMessage(message, attr_name));
  *value = GetValFromPbMessage<T>(message, attr_name);
}

template <typename T>
inline void SetAttr(PbMessage *message, const std::string &attr_name,
                    const T &value) {
  SetValInPbMessage(message, attr_name, value);
}

template <>
inline void GetAttr<Shape>(const PbMessage &message,
                           const std::string &attr_name, Shape *value) {
  CHECK(HasFieldInPbMessage(message, attr_name));
  *value = Shape(GetValFromPbMessage<ShapeProto>(message, attr_name));
}

template <>
inline void SetAttr<Shape>(PbMessage *message, const std::string &attr_name,
                           const Shape &value) {
  ShapeProto shape;
  value.ToProto(&shape);
  SetValInPbMessage<ShapeProto>(message, attr_name, shape);
}

template <typename T>
inline void GetMessage(const PbMessage &message, const std::string &attr_name,
                       T **value) {
  CHECK(HasFieldInPbMessage(message, attr_name));
  *value = dynamic_cast<T *>(const_cast<oneflow::PbMessage *>(
      &GetMessageInPbMessage(message, attr_name)));
}

inline std::string GetAttrAsString(const PbMessage &message,
                                   const std::string &attr_name) {
  std::string value;
  GetAttr<std::string>(message, attr_name, &value);
  return std::move(value);
}

inline bool HasAttr(const PbMessage &message, const std::string &attr_name) {
  using namespace google::protobuf;
  const Descriptor *d = message.GetDescriptor();
  const FieldDescriptor *fd = d->FindFieldByName(attr_name);
  if (fd && fd->is_optional()) {
    const Reflection *r = message.GetReflection();
    return r->HasField(message, fd);
  }
  return fd != nullptr;
}

inline void GetOneofType(const PbMessage &message,
                         const std::string &oneof_name,
                         std::string *oneof_type) {
  using namespace google::protobuf;
  const Descriptor *d = message.GetDescriptor();
  const OneofDescriptor *ofd = d->FindOneofByName(oneof_name);
  const Reflection *r = message.GetReflection();
  CHECK(ofd) << "Message has no oneof field named " << oneof_name;
  const google::protobuf::FieldDescriptor *fd =
      r->GetOneofFieldDescriptor(message, ofd);
  *oneof_type = fd->name();
}

template <typename T>
inline void GetOneofMessage(const PbMessage &message,
                            const std::string &oneof_name, T **value) {
  using namespace google::protobuf;
  const Descriptor *d = message.GetDescriptor();
  const OneofDescriptor *ofd = d->FindOneofByName(oneof_name);
  const Reflection *r = message.GetReflection();
  CHECK(ofd) << "Message has no oneof field named " << oneof_name;
  const google::protobuf::FieldDescriptor *fd =
      r->GetOneofFieldDescriptor(message, ofd);
  *value = dynamic_cast<T *>(
      const_cast<oneflow::PbMessage *>(&(r->GetMessage(message, fd))));
}

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_
