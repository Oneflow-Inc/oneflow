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
#include <memory>
#include <mutex>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/object_pool.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"

namespace oneflow {

class Stream;

namespace vm {

static const int kInstructionStatusBufferBytes = 64;

class InstructionStatusBuffer final {
 public:
  InstructionStatusBuffer() = default;
  ~InstructionStatusBuffer() = default;

  const char* buffer() const { return &buffer_[0]; }
  char* mut_buffer() { return &buffer_[0]; }

 private:
  char buffer_[kInstructionStatusBufferBytes];
};

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

  void __Init__(Stream* stream, std::shared_ptr<InstructionPolicy>&& instruction_policy);

  // Getters
  const Stream& stream() const { return *stream_; }
  const InstructionStatusBuffer& status_buffer() const { return status_buffer_; }
  const intrusive::ListHook& main_instruction_hook() const { return main_instruction_hook_; }
  const InstructionPolicy& instruction_policy() const { return *instruction_policy_; }
  std::string DebugName() const;

  const intrusive::ListHook& dispatched_instruction_hook() const {
    return dispatched_instruction_hook_;
  }
  const intrusive::ListHook& lively_instruction_hook() const { return lively_instruction_hook_; }
  const intrusive::ListHook& worker_pending_instruction_hook() const {
    return worker_pending_instruction_hook_;
  }
  const intrusive::ListHook& barrier_instruction_hook() const { return barrier_instruction_hook_; }
  const InEdgeList& in_edges() const { return in_edges_; }
  const OutEdgeList& out_edges() const { return out_edges_; }
  const DependenceAccessList& access_list() const { return access_list_; }

  Maybe<void> Prepare();
  void Compute();

  // Setters
  Stream* mut_stream() { return stream_; }
  InstructionStatusBuffer* mut_status_buffer() { return &status_buffer_; }
  InstructionPolicy* mut_instruction_policy() { return instruction_policy_.get(); }
  InEdgeList* mut_in_edges() { return &in_edges_; }
  OutEdgeList* mut_out_edges() { return &out_edges_; }
  DependenceAccessList* mut_access_list() { return &access_list_; }

  // methods
  void InitStatus();
  void DeleteStatusAndCheckEdges();
  bool Launched() const;
  bool Done() const;
  StreamPolicy* mut_stream_policy();
  const StreamPolicy& stream_policy() const;
  std::shared_ptr<Frame> foreign_frame() const { return foreign_frame_; }

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

  // used for instructions building, pending to scheduler, constructing DAG, pending to callback
  // thread and so on.
  // lifetime of barrier instructions:
  //
  //   |<-----main_instruction_hook_----->|
  //                                    |<-----------lively_instruction_hook_---------------->|
  //                                          |<---------barrier_instruction_hook_--------->|
  //
  //
  // lifetime of non-barrier instructions:
  //
  //   |<-----main_instruction_hook_----->|
  //                                    |<-----------lively_instruction_hook_---------------->|
  //                                          |<-------dispatched_instruction_hook_-------->|
  //                                               |<--worker_pending_instruction_hook_-->|
  //
  //
  intrusive::ListHook main_instruction_hook_;
  // dispatched to Stream
  intrusive::ListHook dispatched_instruction_hook_;
  // valid during vm processing
  intrusive::ListHook lively_instruction_hook_;
  // pending to ThreadCtx
  intrusive::ListHook worker_pending_instruction_hook_;
  // for barrier instruction
  intrusive::ListHook barrier_instruction_hook_;

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  Instruction()
      : main_instruction_hook_(),
        dispatched_instruction_hook_(),
        lively_instruction_hook_(),
        worker_pending_instruction_hook_(),
        barrier_instruction_hook_(),
        access_list_(),
        in_edges_(),
        out_edges_(),
        intrusive_ref_(),
        stream_(),
        instruction_policy_(),
        status_buffer_() {}

  // lists
  DependenceAccessList access_list_;
  InEdgeList in_edges_;
  OutEdgeList out_edges_;

  // fields
  intrusive::Ref intrusive_ref_;
  Stream* stream_;
  std::shared_ptr<InstructionPolicy> instruction_policy_;
  InstructionStatusBuffer status_buffer_;
  std::shared_ptr<Frame> foreign_frame_;
};

using InstructionList = intrusive::List<INTRUSIVE_FIELD(Instruction, main_instruction_hook_)>;

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION__H_
