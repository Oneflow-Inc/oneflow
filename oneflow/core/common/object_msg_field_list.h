#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_FIELD_LIST_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_FIELD_LIST_H_

#include <type_traits>
#include <typeinfo>
#include "oneflow/core/common/object_msg_field_list.pb.h"

namespace oneflow {

struct ObjectMsgFieldListUtil final {
  static std::string ToDotNode(const std::string& object_msg_type_name,
                               const ObjectMsgFieldList& object_msg_field_list);
  static std::string ToDotEdges(const std::string& object_msg_type_name,
                                const ObjectMsgFieldList& object_msg_field_list);

  template<typename FieldType>
  static void SetPointerRemovedFieldType(ObjectMsgFieldTypeAndName* field) {
    PointerRemovedFieldTypeSetter<FieldType, std::is_pointer<FieldType>::value>::Call(field);
  }

 private:
  template<typename FieldType, bool is_pointer, typename Enable = void>
  struct PointerRemovedFieldTypeSetter final {
    static void Call(ObjectMsgFieldTypeAndName* field) {}
  };
  template<typename FieldType, typename Enable>
  struct PointerRemovedFieldTypeSetter<FieldType, true, Enable> final {
    static void Call(ObjectMsgFieldTypeAndName* field) {
      using type = typename std::remove_pointer<FieldType>::type;
      field->set_pointer_removed_field_type(typeid(type).name());
    }
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_FIELD_LIST_H_
