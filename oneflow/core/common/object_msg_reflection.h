#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_

#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/object_msg_field_list.pb.h"

namespace oneflow {

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

template<typename T>
class ObjectMsgReflection final {
 public:
  void ReflectObjectMsgFields(ObjectMsgFieldList* obj_msg_field_list) {
    T::template __WalkStaticVerboseField__<StaticDumpObjectMsgFieldName>(obj_msg_field_list);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_REFLECTION_H_
