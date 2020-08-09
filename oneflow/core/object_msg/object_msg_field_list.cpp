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
#include <sstream>
#include <cxxabi.h>
#include "oneflow/core/object_msg/object_msg_field_list.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

std::string Demangle(std::string name) {
  // https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
  int status = -4;  // some arbitrary value to eliminate the compiler warning

  // enable c++11 by passing the flag -std=c++11 to g++
  std::unique_ptr<char, void (*)(void*)> res(abi::__cxa_demangle(name.c_str(), NULL, NULL, &status),
                                             std::free);
  if (status == 0) {
    std::string demangled = res.get();
    if (demangled.find("<") == std::string::npos) { name = demangled; }
  }
  return name;
}

}  // namespace

std::string ObjectMsgFieldListUtil::ToDotNode(const std::string& object_msg_type_name,
                                              const ObjectMsgFieldList& object_msg_field_list) {
  std::stringstream ss;
  ss << object_msg_type_name << " [shape=Mrecord label=\"{\n";
  ss << Demangle(object_msg_type_name) << "\n";
  for (const auto& field : object_msg_field_list.object_msg_field()) {
    ss << "|";
    if (field.has_union_field_list()) {
      ss << "{";
      if (field.union_field_list().union_field_size() > 1) {
        ss << " oneof ";
      } else {
        ss << " optional ";
      }
      ss << "|{";
      int counter = 0;
      for (const auto& union_field : field.union_field_list().union_field()) {
        if (counter++ > 0) { ss << "|"; }
        if (union_field.has_pointer_removed_field_type()) {
          ss << "<" << union_field.field_name() << "> " << union_field.field_name();
        } else {
          ss << Demangle(union_field.field_type()) << " " << union_field.field_name();
        }
      }
      ss << "}";
      ss << "}";
    } else if (field.has_struct_field()) {
      ss << "<" << field.struct_field().field_name() << "> " << field.struct_field().field_name();
    } else {
      UNIMPLEMENTED();
    }
  }
  ss << "| " << Demangle(object_msg_type_name) << "\n";
  ss << "\n}\"];\n";
  return ss.str();
}

std::string ObjectMsgFieldListUtil::ToDotEdges(const std::string& object_msg_type_name,
                                               const ObjectMsgFieldList& object_msg_field_list) {
  std::stringstream ss;
  for (const auto& field : object_msg_field_list.object_msg_field()) {
    if (!field.has_union_field_list()) { continue; }
    for (const auto& union_field : field.union_field_list().union_field()) {
      if (!union_field.has_pointer_removed_field_type()) { continue; }
      ss << object_msg_type_name << ":" << union_field.field_name() << " -> "
         << union_field.pointer_removed_field_type() << ";\n";
    }
  }
  return ss.str();
}

}  // namespace oneflow
