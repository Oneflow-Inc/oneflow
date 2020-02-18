#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_

#include <unordered_map>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/object_msg_field_list.pb.h"

namespace oneflow {

template<typename T>
class ObjectMsgReflection final {
 public:
  void ReflectObjectMsgFields(ObjectMsgFieldList* obj_msg_field_list);
  void RecursivelyReflectObjectMsgFields(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list);
};

ObjectMsgUnionFieldList* FindExistedUnionFieldList(ObjectMsgFieldList* obj_msg_field_list,
                                                   const std::string& oneof_name) {
  std::size_t size = obj_msg_field_list->object_msg_field().size();
  if (size == 0) { return nullptr; }
  auto* last = obj_msg_field_list->mutable_object_msg_field(size - 1);
  if (!last->has_union_field_list()) { return nullptr; }
  if (last->union_field_list().union_name() != oneof_name) { return nullptr; }
  return last->mutable_union_field_list();
}

template<int field_counter, typename WalkCtxType, typename FieldType, bool is_oneof_field>
struct StaticDumpObjectMsgFieldName {
  static void Call(ObjectMsgFieldList* obj_msg_field_list, const char* field_name,
                   const char* oneof_name) {
    std::string field_name_str(field_name);
    if (!is_oneof_field) {
      obj_msg_field_list->mutable_object_msg_field()->Add()->set_struct_field_name(field_name_str);
      return;
    }
    std::string oneof_name_str(oneof_name);
    ObjectMsgUnionFieldList* union_field_list =
        FindExistedUnionFieldList(obj_msg_field_list, oneof_name_str);
    if (union_field_list == nullptr) {
      union_field_list =
          obj_msg_field_list->mutable_object_msg_field()->Add()->mutable_union_field_list();
      union_field_list->set_union_name(oneof_name_str);
    }
    union_field_list->add_union_field_name(field_name_str);
  }
};

template<int field_counter, typename WalkCtxType, typename FieldType, bool is_oneof_field>
struct StaticRecursivelyDumpObjectMsgFieldName {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list,
      const char* field_name, const char* oneof_name) {
    // do nothing
  }
};

template<typename FieldType, bool is_obj_msg_ptr>
struct _StaticRecursivelyDumpObjectMsgFieldName {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list) {
    // do nothing
  }
};

template<typename FieldType>
struct _StaticRecursivelyDumpObjectMsgFieldName<FieldType, true> {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list) {
    const auto& map = *mangled_type_name2obj_msg_field_list;
    using ObjectMsgFieldType = typename std::remove_pointer<FieldType>::type;
    if (map.find(typeid(ObjectMsgFieldType).name()) != map.end()) { return; }
    ObjectMsgReflection<ObjectMsgFieldType>().RecursivelyReflectObjectMsgFields(
        mangled_type_name2obj_msg_field_list);
  }
};

template<int field_counter, typename WalkCtxType, typename FieldType>
struct StaticRecursivelyDumpObjectMsgFieldName<field_counter, WalkCtxType, FieldType, true> {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list,
      const char* field_name, const char* oneof_name) {
    _StaticRecursivelyDumpObjectMsgFieldName<FieldType, std::is_pointer<FieldType>::value>::Call(
        mangled_type_name2obj_msg_field_list);
  }
};

template<typename T>
void ObjectMsgReflection<T>::ReflectObjectMsgFields(ObjectMsgFieldList* obj_msg_field_list) {
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgFieldName>(obj_msg_field_list);
}

template<typename T>
void ObjectMsgReflection<T>::RecursivelyReflectObjectMsgFields(
    std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list) {
  auto* obj_msg_field_list = &(*mangled_type_name2obj_msg_field_list)[typeid(T).name()];
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgFieldName>(obj_msg_field_list);
  T::template __WalkStaticVerboseField__<StaticRecursivelyDumpObjectMsgFieldName>(
      mangled_type_name2obj_msg_field_list);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_
