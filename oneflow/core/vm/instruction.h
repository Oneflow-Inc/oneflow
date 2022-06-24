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
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/object_pool.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {

class Stream;

namespace vm {

class InstructionMsg final : public intrusive::Base {
 public:
  // methods
  void __Init__(Stream* stream, const InstructionType* instruction_type,
                const std::shared_ptr<PhyInstrOperand>& phy_instr_operand);

  // Getters
  const Stream& stream() const { return *stream_; }
  Stream* mut_stream() { return stream_; }
  const InstructionType& instruction_type() const { return *instruction_type_; }
  const std::shared_ptr<PhyInstrOperand>& phy_instr_operand() const { return phy_instr_operand_; }

  std::string DebugName() const;

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  InstructionMsg()
      : intrusive_ref_(), stream_(), instruction_type_(), phy_instr_operand_(), instr_msg_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  Stream* stream_;
  const InstructionType* instruction_type_;
  std::shared_ptr<PhyInstrOperand> phy_instr_operand_;

 public:
  // list hooks
  intrusive::ListHook instr_msg_hook_;
};

using InstructionMsgList = intrusive::List<INTRUSIVE_FIELD(InstructionMsg, instr_msg_hook_)>;

static const int kInstructionStatusBufferBytes = 64;

// clang-format off
FLAT_MSG_BEGIN(InstructionStatusBuffer);
  FLAT_MSG_DEFINE_REPEATED(char, buffer, kInstructionStatusBufferBytes);
FLAT_MSG_END(InstructionStatusBuffer);
// clang-format on

class Instruction;
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

class Stream;
class Instruction final : public intrusive::Base {
 public:
  // types
  using InEdgeList = intrusive::List<INTRUSIVE_FIELD(InstructionEdge, in_edge_hook_)>;
  using OutEdgeList = intrusive::List<INTRUSIVE_FIELD(InstructionEdge, out_edge_hook_)>;
  using DependenceAccessList =
      intrusive::List<INTRUSIVE_FIELD(DependenceAccess, instruction_access_hook_)>;

  // Getters
  const Stream& stream() const { return instr_msg_->stream(); }
  const InstructionMsg& instr_msg() const { return instr_msg_.Get(); }
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
  const DependenceAccessList& access_list() const { return access_list_; }

  // Setters
  Stream* mut_stream() { return instr_msg_->mut_stream(); }
  InstructionMsg* mut_instr_msg() { return CHECK_NOTNULL(instr_msg_.Mutable()); }
  void reset_instr_msg(InstructionMsg* instr_msg) { instr_msg_.Reset(instr_msg); }
  void clear_instr_msg() { instr_msg_.Reset(); }
  InstructionStatusBuffer* mut_status_buffer() { return status_buffer_.Mutable(); }
  InEdgeList* mut_in_edges() { return &in_edges_; }
  OutEdgeList* mut_out_edges() { return &out_edges_; }
  DependenceAccessList* mut_access_list() { return &access_list_; }

  // methods
  void Init(InstructionMsg* instr_msg);
  void Delete();
  bool Done() const;
  const StreamType& stream_type() const;

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  Instruction()
      : intrusive_ref_(),
        status_buffer_(),
        instr_msg_(),
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
  // lists
  DependenceAccessList access_list_;
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
