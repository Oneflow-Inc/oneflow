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
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {
namespace vm {

TestResourceDescScope::TestResourceDescScope(int64_t gpu_device_num, int64_t cpu_device_num,
                                             int64_t machine_num) {
  Resource resource;
  resource.set_machine_num(machine_num);
  resource.set_gpu_device_num(gpu_device_num);
  resource.set_cpu_device_num(cpu_device_num);
  Global<ResourceDesc, ForSession>::New(resource);
}

TestResourceDescScope::~TestResourceDescScope() { Global<ResourceDesc, ForSession>::Delete(); }

ObjectMsgPtr<VmResourceDesc> TestUtil::NewVmResourceDesc(int64_t device_num, int64_t machine_num) {
  HashMap<std::string, int64_t> map{{"cpu", device_num}, {"gpu", device_num}};
  return ObjectMsgPtr<VmResourceDesc>::New(machine_num, map);
}

int64_t TestUtil::NewParallelDesc(InstructionMsgList* instr_msg_list, const std::string& device_tag,
                                  const std::string& device_name) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(device_tag);
  parallel_conf.add_device_name(device_name);
  int64_t parallel_desc_symbol_id = IdUtil::NewLogicalSymbolId();
  CHECK_JUST(
      Global<symbol::Storage<ParallelDesc>>::Get()->Add(parallel_desc_symbol_id, parallel_conf));
  instr_msg_list->EmplaceBack(
      NewInstruction("NewParallelDescSymbol")->add_int64_operand(parallel_desc_symbol_id));
  return parallel_desc_symbol_id;
}

int64_t TestUtil::NewObject(InstructionMsgList* instr_msg_list, const std::string& device_tag,
                            const std::string& device_name, int64_t* parallel_desc_symbol_id) {
  *parallel_desc_symbol_id = NewParallelDesc(instr_msg_list, device_tag, device_name);
  int64_t logical_object_id = IdUtil::NewLogicalObjectId();
  instr_msg_list->EmplaceBack(NewInstruction("NewObject")
                                  ->add_parallel_desc(*parallel_desc_symbol_id)
                                  ->add_int64_operand(logical_object_id));
  return logical_object_id;
}

int64_t TestUtil::NewSymbol(InstructionMsgList* instr_msg_list) {
  int64_t symbol_id = IdUtil::NewLogicalSymbolId();
  instr_msg_list->EmplaceBack(NewInstruction("NewSymbol")->add_int64_operand(symbol_id));
  return symbol_id;
}

int64_t TestUtil::NewStringSymbol(InstructionMsgList* instr_msg_list, const std::string& str) {
  int64_t str_id = NewSymbol(instr_msg_list);
  CHECK_JUST(Global<symbol::Storage<StringSymbol>>::Get()->Add(str_id, str));
  instr_msg_list->EmplaceBack(NewInstruction("InitStringSymbol")->add_init_symbol_operand(str_id));
  return str_id;
}

void TestUtil::AddStreamDescByInstrNames(VmDesc* vm_desc,
                                         const std::vector<std::string>& instr_names) {
  TestUtil::AddStreamDescByInstrNames(vm_desc, 1, instr_names);
}

void TestUtil::AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                         const std::vector<std::string>& instr_names) {
  auto Insert = [&](const std::string& instr_name) {
    const auto& stream_type_id = LookupInstrTypeId(instr_name).stream_type_id();
    auto stream_desc = ObjectMsgPtr<StreamDesc>::New(stream_type_id, 1, parallel_num, 1);
    vm_desc->mut_stream_type_id2desc()->Insert(stream_desc.Mutable());
  };
  for (const auto& instr_name : instr_names) {
    Insert(instr_name);
    Insert(std::string("Infer-") + instr_name);
  }
}

}  // namespace vm
}  // namespace oneflow
