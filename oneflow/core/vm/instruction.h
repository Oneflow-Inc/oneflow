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
#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION__H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION__H_

#include <cstring>
#include <mutex>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/object_pool.h"
#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/instruction_operand.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/instruction.cfg.h"

namespace oneflow {
namespace vm {

class InstructionOperandList final : public intrusive::Base {
 public:
  void __Init__() {}
  // Getters
  const std::vector<FlatMsg<InstructionOperand>>& operand() const { return operand_; }
  // Setters
  std::vector<FlatMsg<InstructionOperand>>* mut_operand() { return &operand_; }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  InstructionOperandList() : intrusive_ref_(), operand_() {}
  intrusive::Ref intrusive_ref_;
  std::vector<FlatMsg<InstructionOperand>> operand_;
};

class VirtualMachineEngine;

class InstructionMsg final : public intrusive::Base {
 public:
  // Getters
  bool has_parallel_desc_symbol_id() const { return 0 != parallel_desc_symbol_id_; }
  int64_t parallel_desc_symbol_id() const { return parallel_desc_symbol_id_; }
  const InstructionOperandList& operand_list() const {
    if (operand_list_) { return operand_list_.Get(); }
    static const auto default_val = intrusive::make_shared<InstructionOperandList>();
    return default_val.Get();
  }
  const std::string& instr_type_name() const { return instr_type_name_; }
  const InstrTypeId& instr_type_id() const { return instr_type_id_; }
  const std::shared_ptr<const ParallelDesc>& phy_instr_parallel_desc() const {
    return phy_instr_parallel_desc_;
  }
  const std::shared_ptr<PhyInstrOperand>& phy_instr_operand() const { return phy_instr_operand_; }
  Stream* phy_instr_stream() const { return phy_instr_stream_; }
  // Setters
  void set_parallel_desc_symbol_id(int64_t val) { parallel_desc_symbol_id_ = val; }
  InstructionOperandList* mut_operand_list() {
    if (!operand_list_) { operand_list_ = intrusive::make_shared<InstructionOperandList>(); }
    return operand_list_.Mutable();
  }
  void reset_operand_list(const InstructionOperandList& other) {
    operand_list_.Reset(const_cast<InstructionOperandList*>(&other));
  }
  std::string* mut_instr_type_name() { return &instr_type_name_; }
  InstrTypeId* mut_instr_type_id() { return &instr_type_id_; }

  // methods
  void __Init__();
  void __Init__(const std::string& instr_type_name);
  void __Init__(VirtualMachineEngine* vm, const std::string& instr_type_name,
                const std::shared_ptr<const ParallelDesc>& phy_instr_parallel_desc,
                const std::shared_ptr<PhyInstrOperand>& phy_instr_operand);
  void __Init__(const InstructionProto& proto);
  void __Init__(const cfg::InstructionProto& proto);
  void __Init__(const InstructionMsg& instr_msg);

  void ToProto(InstructionProto* proto) const;
  intrusive::shared_ptr<InstructionMsg> add_parallel_desc(int64_t symbol_id);
  intrusive::shared_ptr<InstructionMsg> add_double_operand(double double_operand);
  intrusive::shared_ptr<InstructionMsg> add_int64_operand(int64_t int64_operand);
  intrusive::shared_ptr<InstructionMsg> add_uint64_operand(uint64_t uint64_operand);
  intrusive::shared_ptr<InstructionMsg> add_bool_operand(bool bool_operand);
  intrusive::shared_ptr<InstructionMsg> add_separator();
  intrusive::shared_ptr<InstructionMsg> add_const_operand(ObjectId logical_object_id);
  intrusive::shared_ptr<InstructionMsg> add_const_operand(ObjectId logical_object_id,
                                                          const SoleMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_const_operand(ObjectId logical_object_id,
                                                          const AllMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_symbol_operand(ObjectId logical_object_id);
  intrusive::shared_ptr<InstructionMsg> add_mut_operand(ObjectId logical_object_id);
  intrusive::shared_ptr<InstructionMsg> add_mut_operand(ObjectId logical_object_id,
                                                        const SoleMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_mut_operand(ObjectId logical_object_id,
                                                        const AllMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_init_symbol_operand(ObjectId logical_object_id);
  intrusive::shared_ptr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id);
  intrusive::shared_ptr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id,
                                                         const SoleMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id,
                                                         const AllMirroredObject&);
  intrusive::shared_ptr<InstructionMsg> add_del_operand(ObjectId logical_object_id);
  const std::vector<FlatMsg<InstructionOperand>>& operand() const {
    return operand_list().operand();
  }
  std::vector<FlatMsg<InstructionOperand>>* mut_operand() {
    return mut_operand_list()->mut_operand();
  }
  intrusive::shared_ptr<InstructionMsg> Clone() const;
  intrusive::shared_ptr<InstructionMsg> MakeInferInstrMsg() const;

 private:
  InstructionOperand* add_instr_operand();
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  InstructionMsg()
      : intrusive_ref_(),
        instr_type_id_(),
        instr_type_name_(),
        parallel_desc_symbol_id_(),
        phy_instr_parallel_desc_(),
        operand_list_(),
        phy_instr_operand_(),
        phy_instr_stream_(),
        instr_msg_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  InstrTypeId instr_type_id_;
  // instr_type_name is a necessary reduandant field for method ToProto
  std::string instr_type_name_;
  int64_t parallel_desc_symbol_id_;
  std::shared_ptr<const ParallelDesc> phy_instr_parallel_desc_;
  intrusive::shared_ptr<InstructionOperandList> operand_list_;
  std::shared_ptr<PhyInstrOperand> phy_instr_operand_;
  Stream* phy_instr_stream_;

 public:
  // list hooks
  intrusive::ListHook instr_msg_hook_;
};

using InstructionMsgList = intrusive::List<INTRUSIVE_FIELD(InstructionMsg, instr_msg_hook_)>;

template<OperandMemZoneModifier mem_zone_modifier>
void CheckOperand(const Operand& operand);

static const int kInstructionStatusBufferBytes = 64;

// clang-format off
FLAT_MSG_BEGIN(InstructionStatusBuffer);
  FLAT_MSG_DEFINE_REPEATED(char, buffer, kInstructionStatusBufferBytes);
FLAT_MSG_END(InstructionStatusBuffer);
// clang-format on

struct Instruction;
class InstructionEdge final
    : public intrusive::Base,
      public intrusive::EnableObjectPool<InstructionEdge,
                                         intrusive::kThreadUnsafeAndDisableDestruct> {
 public:
  InstructionEdge()
      : intrusive_ref_(),
        src_instruction_(),
        dst_instruction_(),
        in_edge_hook_(),
        out_edge_hook_() {}
  void __Init__() {
    clear_src_instruction();
    clear_dst_instruction();
  }
  // Getters
  bool has_src_instruction() const { return src_instruction_ != nullptr; }
  bool has_dst_instruction() const { return dst_instruction_ != nullptr; }
  const Instruction& src_instruction() const { return *src_instruction_; }
  const Instruction& dst_instruction() const { return *dst_instruction_; }
  // Setters
  void set_src_instruction(Instruction* val) { src_instruction_ = val; }
  void set_dst_instruction(Instruction* val) { dst_instruction_ = val; }
  void clear_src_instruction() { src_instruction_ = nullptr; }
  void clear_dst_instruction() { dst_instruction_ = nullptr; }
  Instruction* mut_src_instruction() { return src_instruction_; }
  Instruction* mut_dst_instruction() { return dst_instruction_; }
  // methods
  void __Init__(Instruction* src_instruction, Instruction* dst_instruction) {
    __Init__();
    set_src_instruction(src_instruction);
    set_dst_instruction(dst_instruction);
  }

  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

 private:
  intrusive::Ref intrusive_ref_;
  // fields
  Instruction* src_instruction_;
  Instruction* dst_instruction_;

 public:
  // list hooks
  intrusive::ListHook in_edge_hook_;
  intrusive::ListHook out_edge_hook_;
};

struct Stream;
class Instruction final : public intrusive::Base {
 public:
  // types
  using InEdgeList = intrusive::List<INTRUSIVE_FIELD(InstructionEdge, in_edge_hook_)>;
  using OutEdgeList = intrusive::List<INTRUSIVE_FIELD(InstructionEdge, out_edge_hook_)>;
  using RwMutexedObjectAccessList =
      intrusive::List<INTRUSIVE_FIELD(RwMutexedObjectAccess, instruction_access_hook_)>;
  using MirroredObjectId2RwMutexedObjectAccess =
      intrusive::SkipList<INTRUSIVE_FIELD(RwMutexedObjectAccess, mirrored_object_id_)>;

  // Getters
  void __Init__() { clear_stream(); }
  bool has_stream() const { return stream_ != nullptr; }
  const Stream& stream() const { return *stream_; }
  const InstructionMsg& instr_msg() const {
    if (instr_msg_) { return instr_msg_.Get(); }
    static const auto default_val = intrusive::make_shared<InstructionMsg>();
    return default_val.Get();
  }
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const { return parallel_desc_; }
  const InstructionStatusBuffer& status_buffer() const { return status_buffer_.Get(); }
  const intrusive::ListHook& instruction_hook() const { return instruction_hook_; }
  const intrusive::ListHook& dispatched_instruction_hook() const {
    return dispatched_instruction_hook_;
  }
  const intrusive::ListHook& lively_instruction_hook() const { return lively_instruction_hook_; }
  const intrusive::ListHook& pending_instruction_hook() const { return pending_instruction_hook_; }
  const intrusive::ListHook& barrier_instruction_hook() const { return barrier_instruction_hook_; }
  const InEdgeList& in_edges() const { return in_edges_; }
  const OutEdgeList& out_edges() const { return out_edges_; }
  const RwMutexedObjectAccessList& access_list() const { return access_list_; }
  const MirroredObjectId2RwMutexedObjectAccess& mirrored_object_id2access() const {
    return mirrored_object_id2access_;
  }

  // Setters
  void set_stream(Stream* val) { stream_ = val; }
  void clear_stream() { stream_ = nullptr; }
  Stream* mut_stream() { return stream_; }
  InstructionMsg* mut_instr_msg() {
    if (!instr_msg_) { instr_msg_ = intrusive::make_shared<InstructionMsg>(); }
    return instr_msg_.Mutable();
  }
  void reset_instr_msg(InstructionMsg* instr_msg) { instr_msg_.Reset(instr_msg); }
  void clear_instr_msg() { instr_msg_.Reset(); }
  std::shared_ptr<const ParallelDesc>* mut_parallel_desc() { return &parallel_desc_; }
  InstructionStatusBuffer* mut_status_buffer() { return status_buffer_.Mutable(); }
  InEdgeList* mut_in_edges() { return &in_edges_; }
  OutEdgeList* mut_out_edges() { return &out_edges_; }
  RwMutexedObjectAccessList* mut_access_list() { return &access_list_; }
  MirroredObjectId2RwMutexedObjectAccess* mut_mirrored_object_id2access() {
    return &mirrored_object_id2access_;
  }

  // methods
  void Init(InstructionMsg* instr_msg, Stream* stream,
            const std::shared_ptr<const ParallelDesc>& parallel_desc);
  void Delete();
  bool QueryLaunched() const {
    return stream_type().QueryInstructionStatusLaunched(stream(), status_buffer());
  }
  bool QueryDoneAfterLaunched() const {
    return stream_type().QueryInstructionStatusDoneAfterLaunched(stream(), status_buffer());
  }
  const StreamType& stream_type() const;
  template<OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_type(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  template<OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_value(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }
  template<OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_type(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  template<OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_value(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }
  template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_type<mem_zone_modifier>(operand.operand());
  }
  template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_value<mem_zone_modifier>(operand.operand());
  }
  template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_type<mem_zone_modifier>(operand.operand());
  }
  template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_value<mem_zone_modifier>(operand.operand());
  }
  template<InterpretType interpret_type>
  MirroredObject* MutMirroredObject(const MutOperand& mut_operand) {
    return MirroredObjectUtil<interpret_type>::Mut(this, mut_operand);
  }
  template<InterpretType interpret_type>
  const MirroredObject* GetMirroredObject(const ConstOperand& const_operand) const {
    return MirroredObjectUtil<interpret_type>::Get(*this, const_operand);
  }
  MirroredObject* mut_type_mirrored_object(const MutOperand& mut_operand);
  MirroredObject* mut_value_mirrored_object(const MutOperand& mut_operand);

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  template<int64_t (*TransformLogicalObjectId)(int64_t)>
  MirroredObject* MutMirroredObject(const Operand& operand, int64_t default_global_device_id);
  template<int64_t (*TransformLogicalObjectId)(int64_t)>
  const MirroredObject* GetMirroredObject(const Operand& operand,
                                          int64_t default_global_device_id) const;
  const RwMutexedObject* operand_type(const Operand& operand,
                                      int64_t default_global_device_id) const;
  const RwMutexedObject* operand_value(const Operand& operand,
                                       int64_t default_global_device_id) const;
  RwMutexedObject* mut_operand_type(const Operand& operand, int64_t default_global_device_id);
  RwMutexedObject* mut_operand_value(const Operand& operand, int64_t default_global_device_id);
  MirroredObject* MutMirroredObject(const Operand& operand, int64_t default_global_device_id) {
    return MutMirroredObject<&IdUtil::GetValueId>(operand, default_global_device_id);
  }
  int64_t GetOperandDefaultGlobalDeviceId() const;
  template<InterpretType interpret_type>
  struct MirroredObjectUtil {
    static const MirroredObject* Get(const Instruction&, const ConstOperand&);
    static MirroredObject* Mut(Instruction*, const MutOperand&);
  };

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  Instruction()
      : intrusive_ref_(),
        status_buffer_(),
        instr_msg_(),
        parallel_desc_(),
        stream_(),
        mirrored_object_id2access_(),
        access_list_(),
        in_edges_(),
        out_edges_(),
        instruction_hook_(),
        dispatched_instruction_hook_(),
        lively_instruction_hook_(),
        pending_instruction_hook_(),
        barrier_instruction_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  FlatMsg<InstructionStatusBuffer> status_buffer_;
  intrusive::shared_ptr<InstructionMsg> instr_msg_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
  Stream* stream_;
  // maps
  MirroredObjectId2RwMutexedObjectAccess mirrored_object_id2access_;
  // lists
  RwMutexedObjectAccessList access_list_;
  InEdgeList in_edges_;
  OutEdgeList out_edges_;

 public:
  // pending or waiting list hooks
  intrusive::ListHook instruction_hook_;
  // dispatched to Stream
  intrusive::ListHook dispatched_instruction_hook_;
  // valid during vm processing
  intrusive::ListHook lively_instruction_hook_;
  // pending to ThreadCtx
  intrusive::ListHook pending_instruction_hook_;
  intrusive::ListHook barrier_instruction_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION__H_
