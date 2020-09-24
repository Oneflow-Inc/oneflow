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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_REFLECTION_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_REFLECTION_H_

#include <sstream>
#include <unordered_map>
#include <vector>
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/object_msg/object_msg_field_list.h"

namespace oneflow {

template<typename T>
class ObjectMsgReflection final {
 public:
  void ReflectObjectMsgFields(ObjectMsgFieldList* obj_msg_field_list) const;
  void RecursivelyReflectObjectMsgFields(std::unordered_map<std::string, ObjectMsgFieldList>*
                                             mangled_type_name2obj_msg_field_list) const;
  void ReflectLinkEdges(std::set<ObjectMsgContainerLinkEdge>* edges) const;
  void RecursivelyReflectLinkEdges(std::set<ObjectMsgContainerLinkEdge>* edges) const;
};

template<typename... Args>
class ObjectMsgListReflection final {
 public:
  std::string ToDot() const { return ToDot(""); }
  std::string ToDot(const std::string& digraph_name) const;
};

// details

inline ObjectMsgUnionFieldList* FindExistedUnionFieldList(ObjectMsgFieldList* obj_msg_field_list,
                                                          const std::string& oneof_name) {
  std::size_t size = obj_msg_field_list->object_msg_field().size();
  if (size == 0) { return nullptr; }
  auto* last = obj_msg_field_list->mutable_object_msg_field(size - 1);
  if (!last->has_union_field_list()) { return nullptr; }
  if (last->union_field_list().union_name() != oneof_name) { return nullptr; }
  return last->mutable_union_field_list();
}

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType,
         bool is_oneof_field>
struct StaticDumpObjectMsgFieldName {
  static void Call(ObjectMsgFieldList* obj_msg_field_list, const char* field_name,
                   const char* oneof_name) {
    std::string field_name_str(field_name);
    if (!is_oneof_field) {
      auto* struct_field =
          obj_msg_field_list->mutable_object_msg_field()->Add()->mutable_struct_field();
      struct_field->set_field_type(
          StructT::template __DssFieldTypeId__<field_counter, FieldType>::Call());
      struct_field->set_field_name(field_name_str);
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
    auto* union_field = union_field_list->mutable_union_field()->Add();
    union_field->set_field_type(typeid(FieldType).name());
    union_field->set_field_name(field_name_str);
    ObjectMsgFieldListUtil::SetPointerRemovedFieldType<FieldType>(union_field);
  }
};

template<typename T>
struct StaticRecursivelyDumpObjectMsgContainerElemFieldName final {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list) {
    ObjectMsgReflection<T>().RecursivelyReflectObjectMsgFields(
        mangled_type_name2obj_msg_field_list);
  }
};

template<>
struct StaticRecursivelyDumpObjectMsgContainerElemFieldName<void> final {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list) {}
};

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType,
         bool is_oneof_field>
struct StaticRecursivelyDumpObjectMsgFieldName {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list,
      const char* field_name, const char* oneof_name) {
    using elem_type = typename StructT::template ContainerElemStruct<field_counter>::type;
    StaticRecursivelyDumpObjectMsgContainerElemFieldName<elem_type>::Call(
        mangled_type_name2obj_msg_field_list);
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

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType>
struct StaticRecursivelyDumpObjectMsgFieldName<StructT, field_counter, WalkCtxType, FieldType,
                                               true> {
  static void Call(
      std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list,
      const char* field_name, const char* oneof_name) {
    _StaticRecursivelyDumpObjectMsgFieldName<FieldType, std::is_pointer<FieldType>::value>::Call(
        mangled_type_name2obj_msg_field_list);
  }
};

template<typename T>
void ObjectMsgReflection<T>::ReflectObjectMsgFields(ObjectMsgFieldList* obj_msg_field_list) const {
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgFieldName>(obj_msg_field_list);
}

template<typename T>
void ObjectMsgReflection<T>::RecursivelyReflectObjectMsgFields(
    std::unordered_map<std::string, ObjectMsgFieldList>* mangled_type_name2obj_msg_field_list)
    const {
  const auto& map = *mangled_type_name2obj_msg_field_list;
  if (map.find(typeid(T).name()) != map.end()) { return; }
  auto* obj_msg_field_list = &(*mangled_type_name2obj_msg_field_list)[typeid(T).name()];
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgFieldName>(obj_msg_field_list);
  T::template __WalkStaticVerboseField__<StaticRecursivelyDumpObjectMsgFieldName>(
      mangled_type_name2obj_msg_field_list);
}

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType,
         bool is_oneof>
struct StaticDumpObjectMsgLinkEdges {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges, const char*, const char*) {
    StructT::template LinkEdgesGetter<field_counter>::Call(edges);
  }
};

template<typename T>
struct StaticRecursivelyDumpObjectMsgContainerElemLinkEdges final {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {
    ObjectMsgReflection<T>().RecursivelyReflectLinkEdges(edges);
  }
};

template<>
struct StaticRecursivelyDumpObjectMsgContainerElemLinkEdges<void> final {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {}
};

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType,
         bool is_oneof_field>
struct StaticRecursivelyDumpObjectMsgLinkEdge {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges, const char* field_name,
                   const char* oneof_name) {
    using elem_type = typename StructT::template ContainerElemStruct<field_counter>::type;
    StaticRecursivelyDumpObjectMsgContainerElemLinkEdges<elem_type>::Call(edges);
  }
};

template<typename FieldType, bool is_obj_msg_ptr>
struct _StaticRecursivelyDumpObjectMsgLinkEdge {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {
    // do nothing
  }
};

template<typename FieldType>
struct _StaticRecursivelyDumpObjectMsgLinkEdge<FieldType, true> {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {
    using ObjectMsgFieldType = typename std::remove_pointer<FieldType>::type;
    ObjectMsgReflection<ObjectMsgFieldType>().RecursivelyReflectLinkEdges(edges);
  }
};

template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType>
struct StaticRecursivelyDumpObjectMsgLinkEdge<StructT, field_counter, WalkCtxType, FieldType,
                                              true> {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges, const char* field_name,
                   const char* oneof_name) {
    _StaticRecursivelyDumpObjectMsgLinkEdge<FieldType, std::is_pointer<FieldType>::value>::Call(
        edges);
  }
};
template<typename StructT, int field_counter, typename WalkCtxType, typename FieldType,
         bool is_oneof>
struct RecursivelyStaticDumpObjectMsgLinkEdges {
  static void Call(std::set<ObjectMsgContainerLinkEdge>* edges, const char*, const char*) {
    StructT::template LinkEdgesGetter<field_counter>::Call(edges);
  }
};

template<typename T>
void ObjectMsgReflection<T>::ReflectLinkEdges(std::set<ObjectMsgContainerLinkEdge>* edges) const {
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgLinkEdges>(edges);
}

template<typename T>
void ObjectMsgReflection<T>::RecursivelyReflectLinkEdges(
    std::set<ObjectMsgContainerLinkEdge>* edges) const {
  T::template __WalkStaticVerboseField__<StaticDumpObjectMsgLinkEdges>(edges);
  T::template __WalkStaticVerboseField__<StaticRecursivelyDumpObjectMsgLinkEdge>(edges);
}

template<typename Container>
void ObjectMsgListReflectionObjectMsgFields(Container*) {}

template<typename Container, typename T, typename... Args>
void ObjectMsgListReflectionObjectMsgFields(Container* name2field_list) {
  ObjectMsgReflection<T>().RecursivelyReflectObjectMsgFields(name2field_list);
  ObjectMsgListReflectionObjectMsgFields<Container, Args...>(name2field_list);
}

template<typename Container>
void ObjectMsgListReflectionObjectMsgLinkEdges(Container* edges) {}

template<typename Container, typename T, typename... Args>
void ObjectMsgListReflectionObjectMsgLinkEdges(Container* edges) {
  ObjectMsgReflection<T>().RecursivelyReflectLinkEdges(edges);
  ObjectMsgListReflectionObjectMsgLinkEdges<Container, Args...>(edges);
}

template<typename... Args>
std::string ObjectMsgListReflection<Args...>::ToDot(const std::string& digraph_name) const {
  using Type2FieldList = std::unordered_map<std::string, ObjectMsgFieldList>;
  Type2FieldList type_name2field_list;
  ObjectMsgListReflectionObjectMsgFields<Type2FieldList, Args...>(&type_name2field_list);
  using EdgeVec = std::set<ObjectMsgContainerLinkEdge>;
  EdgeVec link_edges;
  ObjectMsgListReflectionObjectMsgLinkEdges<EdgeVec, Args...>(&link_edges);
  std::stringstream ss;
  ss << "digraph ";
  ss << digraph_name;
  ss << " {\n";
  ss << "node[shape=record];\n";
  for (const auto& pair : type_name2field_list) {
    ss << ObjectMsgFieldListUtil::ToDotNode(pair.first, pair.second) << "\n";
  }
  for (const auto& pair : type_name2field_list) {
    ss << ObjectMsgFieldListUtil::ToDotEdges(pair.first, pair.second) << "\n";
  }
  for (const auto& edge : link_edges) {
    ss << edge.container_type_name << ":" << edge.container_field_name << " -> "
       << edge.elem_type_name << ":" << edge.elem_link_name << "[label=\""
       << edge.container_field_name << "\"]"
       << "\n";
  }
  ss << "}\n";
  return ss.str();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_REFLECTION_H_
