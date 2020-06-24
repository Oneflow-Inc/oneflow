#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

namespace {

template<typename DoEachT>
void ForEachMachineIdAndDeviceIdInRange(const ParallelDesc& parallel_desc,
                                        const Range& machine_id_range, const DoEachT& DoEach) {
  if (machine_id_range.size() < parallel_desc.sorted_machine_ids().size()) {
    FOR_RANGE(int64_t, machine_id, machine_id_range.begin(), machine_id_range.end()) {
      if (parallel_desc.HasMachineId(machine_id)) {
        for (int64_t device_id : parallel_desc.sorted_dev_phy_ids(machine_id)) {
          DoEach(machine_id, device_id);
        }
      }
    }
  } else {
    for (int64_t machine_id : parallel_desc.sorted_machine_ids()) {
      if (machine_id >= machine_id_range.begin() && machine_id < machine_id_range.end()) {
        for (int64_t device_id : parallel_desc.sorted_dev_phy_ids(machine_id)) {
          DoEach(machine_id, device_id);
        }
      }
    }
  }
}

}  // namespace

class NewObjectInstructionType final : public InstructionType {
 public:
  NewObjectInstructionType() = default;
  ~NewObjectInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewObjectInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, logical_object_id);
  FLAT_MSG_VIEW_END(NewObjectInstruction);
  // clang-format on

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<NewObjectInstruction> view(instr_msg->operand());
    std::shared_ptr<ParallelDesc> parallel_desc = vm->GetInstructionParallelDesc(*instr_msg);
    CHECK(static_cast<bool>(parallel_desc));
    const std::shared_ptr<std::string>& device_tag =
        CHECK_JUST(DeviceTag4DeviceType(parallel_desc->device_type()));
    FOR_RANGE(int, i, 0, view->logical_object_id_size()) {
      int64_t logical_object_id = GetLogicalObjectId(view->logical_object_id(i));
      auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(vm->mut_vm_thread_only_allocator(),
                                                                 logical_object_id, parallel_desc);
      CHECK(vm->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      ForEachMachineIdAndDeviceIdInRange(
          *parallel_desc, vm->machine_id_range(), [&](int64_t machine_id, int64_t device_id) {
            int64_t global_device_id =
                vm->vm_resource_desc().GetGlobalDeviceId(machine_id, *device_tag, device_id);
            auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(
                vm->mut_allocator(), logical_object.Mutable(), global_device_id);
            CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
          });
    }
  }
};
COMMAND(RegisterInstructionType<NewObjectInstructionType>("NewObject"));

class BroadcastObjectReferenceInstructionType final : public InstructionType {
 public:
  BroadcastObjectReferenceInstructionType() = default;
  ~BroadcastObjectReferenceInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(BroadcastObjectReferenceInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, new_object);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, sole_mirrored_object);
  FLAT_MSG_VIEW_END(BroadcastObjectReferenceInstruction);
  // clang-format on

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<BroadcastObjectReferenceInstruction> args(instr_msg->operand());
    const RwMutexedObject* sole_rw_mutexed_object = nullptr;
    {
      int64_t object_id = GetLogicalObjectId(args->sole_mirrored_object());
      auto* logical_object = vm->mut_id2logical_object()->FindPtr(object_id);
      CHECK_NOTNULL(logical_object);
      auto* map = logical_object->mut_global_device_id2mirrored_object();
      sole_rw_mutexed_object = &map->Begin()->rw_mutexed_object();
      CHECK_NOTNULL(sole_rw_mutexed_object);
    }
    std::shared_ptr<ParallelDesc> parallel_desc = vm->GetInstructionParallelDesc(*instr_msg);
    CHECK(static_cast<bool>(parallel_desc));
    const std::shared_ptr<std::string>& device_tag =
        CHECK_JUST(DeviceTag4DeviceType(parallel_desc->device_type()));
    int64_t new_object = GetLogicalObjectId(args->new_object());
    auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(vm->mut_vm_thread_only_allocator(),
                                                               new_object, parallel_desc);
    CHECK(vm->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
    auto* global_device_id2mirrored_object = logical_object->mut_global_device_id2mirrored_object();
    ForEachMachineIdAndDeviceIdInRange(
        *parallel_desc, vm->machine_id_range(), [&](int64_t machine_id, int64_t device_id) {
          int64_t global_device_id =
              vm->vm_resource_desc().GetGlobalDeviceId(machine_id, *device_tag, device_id);
          auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(
              vm->mut_allocator(), logical_object.Mutable(), global_device_id);
          mirrored_object->reset_rw_mutexed_object(*sole_rw_mutexed_object);
          CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
        });
  }
};
COMMAND(
    RegisterInstructionType<BroadcastObjectReferenceInstructionType>("BroadcastObjectReference"));

class ReplaceMirroredInstructionType final : public InstructionType {
 public:
  ReplaceMirroredInstructionType() = default;
  ~ReplaceMirroredInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(ReplaceMirroredInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, lhs_object_id);
    FLAT_MSG_VIEW_DEFINE_PATTERN(OperandSeparator, separator);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, rhs_object_id);
  FLAT_MSG_VIEW_END(ReplaceMirroredInstruction);
  // clang-format on

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<ReplaceMirroredInstruction> args(instr_msg->operand());
    auto DoEachRhsObject = [&](MirroredObject* lhs, int64_t global_device_id) {
      CHECK_EQ(lhs->rw_mutexed_object().access_list().size(), 0);
      FOR_RANGE(int, i, 0, args->rhs_object_id_size()) {
        int64_t rhs_object_id = GetLogicalObjectId(args->rhs_object_id(i));
        const auto* rhs = vm->GetMirroredObject(rhs_object_id, global_device_id);
        if (rhs != nullptr) {
          lhs->reset_rw_mutexed_object(rhs->rw_mutexed_object());
          break;
        }
      }
    };
    std::shared_ptr<ParallelDesc> parallel_desc = vm->GetInstructionParallelDesc(*instr_msg);
    CHECK(static_cast<bool>(parallel_desc));
    const std::shared_ptr<std::string>& device_tag =
        CHECK_JUST(DeviceTag4DeviceType(parallel_desc->device_type()));
    ForEachMachineIdAndDeviceIdInRange(
        *parallel_desc, vm->machine_id_range(), [&](int64_t machine_id, int64_t device_id) {
          FOR_RANGE(int, i, 0, args->lhs_object_id_size()) {
            int64_t global_device_id =
                vm->vm_resource_desc().GetGlobalDeviceId(machine_id, *device_tag, device_id);
            int64_t lhs_object_id = GetLogicalObjectId(args->lhs_object_id(i));
            auto* lhs = vm->MutMirroredObject(lhs_object_id, global_device_id);
            if (lhs != nullptr) { DoEachRhsObject(lhs, global_device_id); }
          }
        });
  }
};
COMMAND(RegisterInstructionType<ReplaceMirroredInstructionType>("ReplaceMirrored"));

class DeleteObjectInstructionType final : public InstructionType {
 public:
  DeleteObjectInstructionType() = default;
  ~DeleteObjectInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteObjectInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(MutOperand, object);
  FLAT_MSG_VIEW_END(DeleteObjectInstruction);
  // clang-format on

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    // do nothing, delete objects in Compute method
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<DeleteObjectInstruction> view(instr_msg->operand());
    FOR_RANGE(int, i, 0, view->object_size()) {
      CHECK(view->object(i).operand().has_all_mirrored_object());
      int64_t logical_object_id = view->object(i).operand().logical_object_id();
      logical_object_id = GetLogicalObjectId(logical_object_id);
      auto* logical_object = vm->mut_id2logical_object()->FindPtr(logical_object_id);
      CHECK_NOTNULL(logical_object);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      OBJECT_MSG_MAP_FOR_EACH_PTR(global_device_id2mirrored_object, mirrored_object) {
        if (mirrored_object->rw_mutexed_object().ref_cnt() == 1) {
          CHECK(!mirrored_object->rw_mutexed_object().has_object());
        }
        global_device_id2mirrored_object->Erase(mirrored_object);
      }
      vm->mut_id2logical_object()->Erase(logical_object);
    }
  }
};
COMMAND(RegisterInstructionType<DeleteObjectInstructionType>("DeleteObject"));

}  // namespace vm
}  // namespace oneflow
