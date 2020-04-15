#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

TestResourceDescScope::TestResourceDescScope(int64_t gpu_device_num, int64_t cpu_device_num) {
  Resource resource;
  resource.set_machine_num(1);
  resource.set_gpu_device_num(gpu_device_num);
  resource.set_cpu_device_num(cpu_device_num);
  Global<ResourceDesc>::New(resource);
}

TestResourceDescScope::~TestResourceDescScope() { Global<ResourceDesc>::Delete(); }

ObjectMsgPtr<VmResourceDesc> TestUtil::NewVmResourceDesc(int64_t device_num, int64_t machine_num) {
  HashMap<std::string, int64_t> map{{"cpu", device_num}, {"gpu", device_num}};
  return ObjectMsgPtr<VmResourceDesc>::New(machine_num, map);
}

int64_t TestUtil::NewObject(InstructionMsgList* instr_msg_list, const std::string& device_name) {
  auto parallel_conf = std::make_shared<ParallelConf>();
  parallel_conf->add_device_name(device_name);
  int64_t parallel_conf_logical_object_id = NewConstHostLogicalObjectId();
  Global<Storage<ParallelConf>>::Get()->Add(parallel_conf_logical_object_id, parallel_conf);
  instr_msg_list->EmplaceBack(
      NewInstruction("NewParallelDescSymbol")->add_int64_operand(parallel_conf_logical_object_id));
  int64_t logical_object_id = NewNaiveLogicalObjectId();
  instr_msg_list->EmplaceBack(NewInstruction("NewObject")
                                  ->add_int64_operand(parallel_conf_logical_object_id)
                                  ->add_int64_operand(logical_object_id));
  return logical_object_id;
}

int64_t TestUtil::NewSymbol(InstructionMsgList* instr_msg_list) {
  int64_t symbol_value = vm::NewConstHostLogicalObjectId();
  instr_msg_list->EmplaceBack(
      NewInstruction("NewConstHostSymbol")->add_int64_operand(symbol_value));
  return symbol_value;
}

int64_t TestUtil::NewStringSymbol(InstructionMsgList* instr_msg_list, const std::string& str) {
  int64_t str_id = NewSymbol(instr_msg_list);
  Global<Storage<std::string>>::Get()->Add(str_id, std::make_shared<std::string>(str));
  instr_msg_list->EmplaceBack(
      NewInstruction("InitStringSymbol")->add_init_const_host_operand(str_id));
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
    auto instr_type = ObjectMsgPtr<StreamDesc>::New(stream_type_id, 1, parallel_num, 1);
    vm_desc->mut_stream_type_id2desc()->Insert(instr_type.Mutable());
  };
  for (const auto& instr_name : instr_names) {
    Insert(instr_name);
    Insert(std::string("Infer-") + instr_name);
  }
}

}  // namespace vm
}  // namespace oneflow
