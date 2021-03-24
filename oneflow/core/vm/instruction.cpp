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
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

template<InterpretType interpret_type>
int64_t GetObjectId(int64_t);

template<>
int64_t GetObjectId<kCompute>(int64_t val) {
  return IdUtil::GetValueId(val);
}

template<>
int64_t GetObjectId<kInfer>(int64_t val) {
  return IdUtil::GetTypeId(val);
}

template<typename T>
void InitFromProto(InstructionMsg* that, const T& proto) {
  that->__Init__(proto.instr_type_name());
  that->mutable_operand()->resize(proto.operand_size());
  if (proto.has_parallel_desc_symbol_id()) {
    that->set_parallel_desc_symbol_id(proto.parallel_desc_symbol_id());
  }
  for (int i = 0; i < proto.operand_size(); ++i) {
    that->mutable_operand()->at(i)->__Init__(proto.operand(i));
  }
}

}  // namespace

InstructionOperand* InstructionMsg::add_instr_operand() {
  auto* operand_vec = mutable_operand();
  operand_vec->emplace_back();
  return operand_vec->back().Mutable();
}

void InstructionMsg::__Init__() {
  *mutable_instr_type_name() = "";
  mutable_operand_list();
}

void InstructionMsg::__Init__(const std::string& instr_type_name) {
  __Init__();
  mutable_instr_type_id()->CopyFrom(LookupInstrTypeId(instr_type_name));
  *mutable_instr_type_name() = instr_type_name;
}

void InstructionMsg::__Init__(const InstructionProto& proto) { InitFromProto(this, proto); }
void InstructionMsg::__Init__(const cfg::InstructionProto& proto) { InitFromProto(this, proto); }

void InstructionMsg::__Init__(const InstructionMsg& instr_msg) {
  __Init__();
  mutable_instr_type_id()->CopyFrom(instr_msg.instr_type_id());
  *mutable_instr_type_name() = instr_msg.instr_type_name();
  if (instr_msg.has_parallel_desc_symbol_id()) {
    set_parallel_desc_symbol_id(instr_msg.parallel_desc_symbol_id());
  }
  reset_operand_list(instr_msg.operand_list());
}

void InstructionMsg::ToProto(InstructionProto* proto) const {
  proto->set_instr_type_name(instr_type_name());
  if (has_parallel_desc_symbol_id()) {
    proto->set_parallel_desc_symbol_id(parallel_desc_symbol_id());
  }
  proto->mutable_operand()->Clear();
  for (const auto& operand : operand_list().operand()) {
    operand->ToProto(proto->mutable_operand()->Add());
  }
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_parallel_desc(int64_t symbol_id) {
  set_parallel_desc_symbol_id(symbol_id);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_double_operand(double double_operand) {
  add_instr_operand()->set_double_operand(double_operand);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_int64_operand(int64_t int64_operand) {
  add_instr_operand()->set_int64_operand(int64_operand);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_uint64_operand(uint64_t uint64_operand) {
  add_instr_operand()->set_uint64_operand(uint64_operand);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_bool_operand(bool bool_operand) {
  add_instr_operand()->set_bool_operand(bool_operand);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_separator() {
  add_instr_operand()->mutable_separator();
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_const_operand(ObjectId logical_object_id) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_const_operand()->mutable_operand()->__Init__(logical_object_id);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_const_operand(
    ObjectId logical_object_id, const SoleMirroredObject& sole_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_const_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                            sole_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_const_operand(
    ObjectId logical_object_id, const AllMirroredObject& all_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_const_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                            all_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_symbol_operand(ObjectId logical_object_id) {
  CHECK(IdUtil::IsSymbolId(logical_object_id));
  add_instr_operand()->mutable_symbol_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                             SoleMirroredObject());
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut_operand(ObjectId logical_object_id) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut_operand()->mutable_operand()->__Init__(logical_object_id);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut_operand(
    ObjectId logical_object_id, const SoleMirroredObject& sole_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                          sole_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut_operand(
    ObjectId logical_object_id, const AllMirroredObject& all_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                          all_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_init_symbol_operand(ObjectId logical_object_id) {
  CHECK(IdUtil::IsSymbolId(logical_object_id));
  add_instr_operand()->mutable_init_symbol_operand()->mutable_operand()->__Init__(
      logical_object_id, SoleMirroredObject());
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut2_operand(ObjectId logical_object_id) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut2_operand()->mutable_operand()->__Init__(logical_object_id);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut2_operand(
    ObjectId logical_object_id, const SoleMirroredObject& sole_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut2_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                           sole_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::add_mut2_operand(
    ObjectId logical_object_id, const AllMirroredObject& all_mirrored_object) {
  CHECK(IdUtil::IsObjectId(logical_object_id));
  add_instr_operand()->mutable_mut2_operand()->mutable_operand()->__Init__(logical_object_id,
                                                                           all_mirrored_object);
  return this;
}

ObjectMsgPtr<InstructionMsg> InstructionMsg::MakeInferInstrMsg() const {
  auto infer_instr_msg = ObjectMsgPtr<InstructionMsg>::NewFrom(mut_allocator(), *this);
  auto* stream_type_id = infer_instr_msg->mut_instr_type_id()->mut_stream_type_id();
  CHECK_EQ(stream_type_id->interpret_type(), InterpretType::kCompute);
  stream_type_id->CopyFrom(LookupInferStreamTypeId(*stream_type_id));
  return infer_instr_msg;
}

template<>
void CheckOperand<kHostConstMemZoneModifier>(const Operand& operand) {
  CHECK(operand.has_sole_mirrored_object());
  CHECK(IdUtil::IsSymbolId(operand.logical_object_id()));
}

template<>
void CheckOperand<kDeviceMemZoneModifier>(const Operand& operand) {
  CHECK(IdUtil::IsObjectId(operand.logical_object_id()));
}

const RwMutexedObject* Instruction::operand_type(const Operand& operand,
                                                 int64_t default_global_device_id) const {
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  const auto* mirrored_object =
      GetMirroredObject<&IdUtil::GetTypeId>(operand, default_global_device_id);
  if (mirrored_object == nullptr) { return nullptr; }
  return &mirrored_object->rw_mutexed_object();
}

const RwMutexedObject* Instruction::operand_value(const Operand& operand,
                                                  int64_t default_global_device_id) const {
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  CHECK_EQ(instr_msg().instr_type_id().stream_type_id().interpret_type(), InterpretType::kCompute);
  const auto* mirrored_object =
      GetMirroredObject<&IdUtil::GetValueId>(operand, default_global_device_id);
  if (mirrored_object == nullptr) { return nullptr; }
  return &mirrored_object->rw_mutexed_object();
}

RwMutexedObject* Instruction::mut_operand_type(const Operand& operand,
                                               int64_t default_global_device_id) {
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  return MutMirroredObject<&IdUtil::GetTypeId>(operand, default_global_device_id)
      ->mut_rw_mutexed_object();
}

RwMutexedObject* Instruction::mut_operand_value(const Operand& operand,
                                                int64_t default_global_device_id) {
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  CHECK_EQ(instr_msg().instr_type_id().stream_type_id().interpret_type(), InterpretType::kCompute);
  return MutMirroredObject<&IdUtil::GetValueId>(operand, default_global_device_id)
      ->mut_rw_mutexed_object();
}

template<InterpretType interpret_type>
const MirroredObject* Instruction::MirroredObjectUtil<interpret_type>::Get(
    const Instruction& instruction, const ConstOperand& const_operand) {
  const auto& operand = const_operand.operand();
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  int64_t default_device_id = instruction.GetOperandDefaultGlobalDeviceId();
  return instruction.GetMirroredObject<&GetObjectId<interpret_type>>(operand, default_device_id);
}

template<InterpretType interpret_type>
MirroredObject* Instruction::MirroredObjectUtil<interpret_type>::Mut(
    Instruction* instruction, const MutOperand& mut_operand) {
  const auto& operand = mut_operand.operand();
  CHECK(IdUtil::IsValueId(operand.logical_object_id()));
  int64_t default_device_id = instruction->GetOperandDefaultGlobalDeviceId();
  return instruction->MutMirroredObject<&GetObjectId<interpret_type>>(operand, default_device_id);
}

MirroredObject* Instruction::mut_type_mirrored_object(const MutOperand& mut_operand) {
  return MutMirroredObject<kInfer>(mut_operand);
}
MirroredObject* Instruction::mut_value_mirrored_object(const MutOperand& mut_operand) {
  return MutMirroredObject<kCompute>(mut_operand);
}

template<int64_t (*TransformLogicalObjectId)(int64_t)>
MirroredObject* Instruction::MutMirroredObject(const Operand& operand,
                                               int64_t default_global_device_id) {
  FlatMsg<MirroredObjectId> mirrored_object_id;
  mirrored_object_id->__Init__<TransformLogicalObjectId>(operand, default_global_device_id);
  auto* access = mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
  if (access == nullptr) { return nullptr; }
  return access->mut_mirrored_object();
}

template<int64_t (*TransformLogicalObjectId)(int64_t)>
const MirroredObject* Instruction::GetMirroredObject(const Operand& operand,
                                                     int64_t default_global_device_id) const {
  FlatMsg<MirroredObjectId> mirrored_object_id;
  mirrored_object_id->__Init__<TransformLogicalObjectId>(operand, default_global_device_id);
  const auto* access = mirrored_object_id2access().FindPtr(mirrored_object_id.Get());
  if (access == nullptr) { return nullptr; }
  return &access->mirrored_object();
}

int64_t Instruction::GetOperandDefaultGlobalDeviceId() const { return stream().global_device_id(); }

void Instruction::__Init__(InstructionMsg* instr_msg, Stream* stream,
                           const std::shared_ptr<ParallelDesc>& parallel_desc) {
  mutable_status_buffer();
  reset_instr_msg(instr_msg);
  set_stream(stream);
  stream_type().InitInstructionStatus(*stream, mutable_status_buffer());
  *mutable_parallel_desc() = parallel_desc;
}

void Instruction::__Delete__() {
  stream_type().DeleteInstructionStatus(stream(), mut_status_buffer());
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool Instruction::Done() const {
  return stream_type().QueryInstructionStatusDone(stream(), status_buffer());
}

const StreamType& Instruction::stream_type() const { return stream().stream_type(); }

}  // namespace vm
}  // namespace oneflow
