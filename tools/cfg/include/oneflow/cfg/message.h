#ifndef ONEFLOW_CFG_CFG_MESSAGE_H_
#define ONEFLOW_CFG_CFG_MESSAGE_H_

#include <set>
#include <typeinfo>
#include <typeindex>

namespace google {
namespace protobuf {

class Message;

}
}

namespace oneflow {
namespace cfg {


class Message {
 public:
  Message() = default;
  virtual ~Message() = default;

  // Returns nullptr if field not exists or T is field type.
  template<typename T>
  const T* FieldPtr4FieldName(const std::string& field_name) const {
    return FieldPtr4FieldNumber<T>(FieldNumber4FieldName(field_name));
  }

  // Returns nullptr if field not exists or T is field type.
  template<typename T>
  T* MutableFieldPtr4FieldName(const std::string& field_name) {
    return MutableFieldPtr4FieldNumber<T>(FieldNumber4FieldName(field_name));
  }

  // Returns true if field_name defined.
  // This is nothing related to has_xxx().
  template<typename T>
  bool FieldDefined4FieldName(const std::string& field_name) const {
    int field_number = FieldNumber4FieldName(field_name);
    const auto& type_indices = ValidTypeIndices4FieldNumber(field_number);
    return FieldDefined4FieldNumber(field_number) && type_indices.count(typeid(T)) > 0;
  }

  bool FieldDefined4FieldName(const std::string& field_name) const {
    return FieldDefined4FieldNumber(FieldNumber4FieldName(field_name));
  }

  template<typename T>
  const T* FieldPtr4FieldNumber(int field_number) const {
    const auto& type_indices = ValidTypeIndices4FieldNumber(field_number);
    if (type_indices.count(typeid(T)) == 0) { return nullptr; }
    const void* void_ptr = FieldPtr4FieldNumber(field_number);
    if (void_ptr == nullptr) { return nullptr; }
    const T* __attribute__((__may_alias__)) ptr = reinterpret_cast<const T*>(void_ptr);
    return ptr;
  }

  template<typename T>
  T* MutableFieldPtr4FieldNumber(int field_number) {
    const auto& type_indices = ValidTypeIndices4FieldNumber(field_number);
    if (type_indices.count(typeid(T)) == 0) { return nullptr; }
    void* void_ptr = MutableFieldPtr4FieldNumber(field_number);
    if (void_ptr == nullptr) { return nullptr; }
    T* __attribute__((__may_alias__)) ptr = reinterpret_cast<T*>(void_ptr);
    return ptr;
  }

  virtual int FieldNumber4FieldName(const std::string& field_name) const = 0;
  virtual bool FieldDefined4FieldNumber(int field_number) const = 0; 
  virtual const std::set<std::type_index>& ValidTypeIndices4FieldNumber(int field_number) const = 0;
  virtual const void* FieldPtr4FieldNumber(int field_number) const = 0;
  virtual void* MutableFieldPtr4FieldNumber(int field_number) { return nullptr; }
  
  using PbMessage = ::google::protobuf::Message;
  virtual void ToProto(PbMessage*) const = 0;
  virtual void InitFromProto(const PbMessage&) {};

};

}
}

#endif  // ONEFLOW_CFG_CFG_MESSAGE_H_
