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
#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_

#include <cstdint>
#include "oneflow/core/object_msg/flat_msg.h"

namespace oneflow {
namespace vm {

using ObjectId = int64_t;

struct IdUtil final {
  // usually [-4096, 4096]
  static int64_t IsErrorId(int64_t id);

  static int64_t IsLogicalId(int64_t id) { return IsLogicalValueId(id); }
  static int64_t NewLogicalObjectId() { return NewLogicalValueObjectId(); }
  static int64_t NewLogicalSymbolId() { return NewLogicalValueSymbolId(); }
  static int64_t NewPhysicalObjectId(int32_t machine_id) {
    return NewPhysicalValueObjectId(machine_id);
  }
  static int64_t NewPhysicalSymbolId(int32_t machine_id) {
    return NewPhysicalValueSymbolId(machine_id);
  }

  static int64_t IsLogicalValueId(int64_t id);
  static int64_t NewLogicalValueObjectId();
  static int64_t NewLogicalValueSymbolId();
  static int64_t NewPhysicalValueObjectId(int32_t machine_id);
  static int64_t NewPhysicalValueSymbolId(int32_t machine_id);

  // type object id or value object id
  static bool IsObjectId(int64_t object_id);
  // type symbol id or value symbol id
  static bool IsSymbolId(int64_t symbol_id);

  // type object id or type symbol id
  static int64_t GetTypeId(int64_t id);
  static bool IsTypeId(int64_t id);

  // value object id or value symbol id
  static int64_t GetValueId(int64_t id);
  static bool IsValueId(int64_t id);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
