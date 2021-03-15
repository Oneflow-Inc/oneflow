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
#ifndef ONEFLOW_CORE_VM_TEST_UTIL_H_
#define ONEFLOW_CORE_VM_TEST_UTIL_H_

#include <vector>
#include <string>
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"
#include "oneflow/core/operator/op_node_signature_desc.h"

namespace oneflow {
namespace vm {

class VmDesc;
class VmResourceDesc;

class TestResourceDescScope final {
 public:
  TestResourceDescScope(const TestResourceDescScope&) = delete;
  TestResourceDescScope(TestResourceDescScope&&) = delete;
  TestResourceDescScope(int64_t gpu_device_num, int64_t cpu_device_num, int64_t machine_num);
  TestResourceDescScope(int64_t gpu_device_num, int64_t cpu_device_num)
      : TestResourceDescScope(gpu_device_num, cpu_device_num, 1) {}
  ~TestResourceDescScope();
};

struct TestUtil {
  using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc() { return NewVmResourceDesc(1); }

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num) {
    return NewVmResourceDesc(device_num, 1);
  }

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num, int64_t machine_num);

  // return logical_object_id
  static int64_t NewObject(InstructionMsgList* list, const std::string& device_tag,
                           const std::string& device_name) {
    int64_t parallel_desc_symbol_id = 0;
    return NewObject(list, device_tag, device_name, &parallel_desc_symbol_id);
  }
  static int64_t NewParallelDesc(InstructionMsgList*, const std::string& device_tag,
                                 const std::string& device_name);
  static int64_t NewObject(InstructionMsgList*, const std::string& device_tag,
                           const std::string& device_name, int64_t* parallel_desc_symbol_id);
  static int64_t NewSymbol(InstructionMsgList*);
  static int64_t NewStringSymbol(InstructionMsgList*, const std::string& str);

  static void AddStreamDescByInstrNames(VmDesc* vm_desc,
                                        const std::vector<std::string>& instr_names);
  static void AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                        const std::vector<std::string>& instr_names);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TEST_UTIL_H_
