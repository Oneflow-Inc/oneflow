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
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/instruction.cfg.h"

namespace oneflow {
namespace vm {

namespace {
template<typename T>
void InitFromProto(InstructionOperand* that, const T& proto) {
  if (proto.has_const_operand()) {
    that->mutable_const_operand()->mutable_operand()->__Init__(proto.const_operand());
  } else if (proto.has_mut_operand()) {
    that->mutable_mut_operand()->mutable_operand()->__Init__(proto.mut_operand());
  } else if (proto.has_mut2_operand()) {
    that->mutable_mut2_operand()->mutable_operand()->__Init__(proto.mut2_operand());
  } else if (proto.has_del_operand()) {
    that->mutable_del_operand()->mutable_operand()->__Init__(proto.del_operand());
  } else if (proto.has_symbol_operand()) {
    that->mutable_symbol_operand()->mutable_operand()->__Init__(proto.symbol_operand());
  } else if (proto.has_init_symbol_operand()) {
    that->mutable_init_symbol_operand()->mutable_operand()->__Init__(proto.init_symbol_operand());
  } else if (proto.has_separator()) {
    that->mutable_separator();
  } else if (proto.has_double_operand()) {
    that->set_double_operand(proto.double_operand());
  } else if (proto.has_int64_operand()) {
    that->set_int64_operand(proto.int64_operand());
  } else if (proto.has_uint64_operand()) {
    that->set_uint64_operand(proto.uint64_operand());
  } else if (proto.has_bool_operand()) {
    that->set_bool_operand(proto.bool_operand());
  } else {
    UNIMPLEMENTED();
  }
}
}  // namespace

void InstructionOperand::__Init__(const InstructionOperandProto& proto) {
  InitFromProto(this, proto);
}
void InstructionOperand::__Init__(const cfg::InstructionOperandProto& proto) {
  InitFromProto(this, proto);
}

void InstructionOperand::ToProto(InstructionOperandProto* proto) const {
  if (has_const_operand()) {
    const_operand().operand().ToProto(proto->mutable_const_operand());
  } else if (has_mut_operand()) {
    mut_operand().operand().ToProto(proto->mutable_mut_operand());
  } else if (has_mut2_operand()) {
    mut2_operand().operand().ToProto(proto->mutable_mut2_operand());
  } else if (has_del_operand()) {
    del_operand().operand().ToProto(proto->mutable_del_operand());
  } else if (has_symbol_operand()) {
    symbol_operand().operand().ToProto(proto->mutable_symbol_operand());
  } else if (has_init_symbol_operand()) {
    init_symbol_operand().operand().ToProto(proto->mutable_init_symbol_operand());
  } else if (has_separator()) {
    proto->mutable_separator();
  } else if (has_double_operand()) {
    proto->set_double_operand(double_operand());
  } else if (has_int64_operand()) {
    proto->set_int64_operand(int64_operand());
  } else if (has_uint64_operand()) {
    proto->set_uint64_operand(uint64_operand());
  } else if (has_bool_operand()) {
    proto->set_bool_operand(bool_operand());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace vm
}  // namespace oneflow
