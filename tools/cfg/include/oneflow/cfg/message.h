#ifndef ONEFLOW_CFG_CFG_MESSAGE_H_
#define ONEFLOW_CFG_CFG_MESSAGE_H_

#include <typeinfo>
#include <typeindex>

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
  bool HasField4FieldName(const std::string& field_name) const {
    return HasField4FieldNumber(FieldNumber4FieldName(field_name));
  }

  template<typename T>
  const T* FieldPtr4FieldNumber(int field_number) const {
    const auto& type_index = TypeIndex4FieldNumber(field_number);
    if (type_index != typeid(T)) { return nullptr; }
    const void* void_ptr = FieldPtr4FieldNumber(field_number);
    if (void_ptr == nullptr) { return nullptr; }
    const T* __attribute__((__may_alias__)) ptr = reinterpret_cast<const T*>(void_ptr);
    return ptr;
  }

  template<typename T>
  T* MutableFieldPtr4FieldNumber(int field_number) {
    const auto& type_index = TypeIndex4FieldNumber(field_number);
    if (type_index != typeid(T)) { return nullptr; }
    void* void_ptr = MutableFieldPtr4FieldNumber(field_number);
    if (void_ptr == nullptr) { return nullptr; }
    T* __attribute__((__may_alias__)) ptr = reinterpret_cast<T*>(void_ptr);
    return ptr;
  }

  virtual int FieldNumber4FieldName(const std::string& field_name) const = 0;
  virtual bool HasField4FieldNumber(int field_number) const = 0; 
  virtual const std::type_index& TypeIndex4FieldNumber(int field_number) const = 0;
  virtual const void* FieldPtr4FieldNumber(int field_number) const = 0;
  virtual void* MutableFieldPtr4FieldNumber(int field_number) { return nullptr; }

  struct UndefinedField {};
};

}
}

#endif  // ONEFLOW_CFG_CFG_MESSAGE_H_
