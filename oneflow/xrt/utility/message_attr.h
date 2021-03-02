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
#ifndef ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_
#define ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_

#include "glog/logging.h"

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {
namespace xrt {
namespace util {

template<typename T>
inline void Attr(const PbMessage &message, const std::string &attr_name, T *value) {
  const UserOpConf* user_conf = dynamic_cast<const UserOpConf*>(&message);
  if (user_conf) {
    CHECK(user_conf->attr().find(attr_name) != user_conf->attr().end());
    auto val = user_conf->attr().at(attr_name);
    *value = user_op::AttrValueAccessor<T>::Attr(val);
  } else {
    CHECK(FieldDefinedInPbMessage(message, attr_name));
    *value = GetValFromPbMessage<T>(message, attr_name);
  }
}

template<typename T>
inline void Attr(PbMessage *message, const std::string &attr_name, const T &value) {
  SetValInPbMessage(message, attr_name, value);
}

template<>
inline void Attr<Shape>(const PbMessage &message, const std::string &attr_name, Shape *value) {
  CHECK(FieldDefinedInPbMessage(message, attr_name));
  *value = Shape(GetValFromPbMessage<ShapeProto>(message, attr_name));
}

template<>
inline void Attr<Shape>(PbMessage *message, const std::string &attr_name, const Shape &value) {
  ShapeProto shape;
  value.ToProto(&shape);
  SetValInPbMessage<ShapeProto>(message, attr_name, shape);
}

template<typename T>
inline void GetMessage(const PbMessage &message, const std::string &attr_name, T **value) {
  CHECK(FieldDefinedInPbMessage(message, attr_name));
  *value = dynamic_cast<T *>(
      const_cast<oneflow::PbMessage *>(&GetMessageInPbMessage(message, attr_name)));
}

inline std::string GetAttrAsString(const PbMessage &message, const std::string &attr_name) {
  std::string value;
  Attr<std::string>(message, attr_name, &value);
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

inline void GetOneofType(const PbMessage &message, const std::string &oneof_name,
                         std::string *oneof_type) {
  using namespace google::protobuf;
  const Descriptor *d = message.GetDescriptor();
  const OneofDescriptor *ofd = d->FindOneofByName(oneof_name);
  const Reflection *r = message.GetReflection();
  CHECK(ofd) << "Message has no oneof field named " << oneof_name;
  const google::protobuf::FieldDescriptor *fd = r->GetOneofFieldDescriptor(message, ofd);
  *oneof_type = fd->name();
}

template<typename T>
inline void GetOneofMessage(const PbMessage &message, const std::string &oneof_name, T **value) {
  using namespace google::protobuf;
  const Descriptor *d = message.GetDescriptor();
  const OneofDescriptor *ofd = d->FindOneofByName(oneof_name);
  const Reflection *r = message.GetReflection();
  CHECK(ofd) << "Message has no oneof field named " << oneof_name;
  const google::protobuf::FieldDescriptor *fd = r->GetOneofFieldDescriptor(message, ofd);
  *value = dynamic_cast<T *>(const_cast<oneflow::PbMessage *>(&(r->GetMessage(message, fd))));
}

class MessageAttr {
 public:
  explicit MessageAttr(const PbMessage &message) : message_(message) {}

  const PbMessage &message() const { return message_; }

  template<typename T>
  T Attr(const std::string &attr_name) const {
    T value;
    util::Attr<T>(message_, attr_name, &value);
    return std::move(value);
  }

  template<typename T>
  void Attr(const std::string &attr_name, const T &value) {
    util::Attr<T>(const_cast<PbMessage *>(&message_), attr_name, value);
  }

  bool HasAttr(const std::string &attr_name) const { return util::HasAttr(message_, attr_name); }

  std::string GetOneofType(const std::string &oneof_name) const {
    std::string oneof_type;
    util::GetOneofType(message_, oneof_name, &oneof_type);
    return std::move(oneof_type);
  }

 private:
  const PbMessage &message_;
};

template<>
inline PbMessage *MessageAttr::Attr<PbMessage *>(const std::string &attr_name) const {
  PbMessage *value = nullptr;
  GetMessage(message_, attr_name, &value);
  return value;
}

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_MESSAGE_ATTR_H_
