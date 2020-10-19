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

namespace oneflow {
namespace vm {

void InstructionOperand::__Init__(const InstructionOperandProto& proto) {
  if (proto.has_const_operand()) {
    mutable_const_operand()->mutable_operand()->__Init__(proto.const_operand());
  } else if (proto.has_mut_operand()) {
    mutable_mut_operand()->mutable_operand()->__Init__(proto.mut_operand());
  } else if (proto.has_mut2_operand()) {
    mutable_mut2_operand()->mutable_operand()->__Init__(proto.mut2_operand());
  } else if (proto.has_symbol_operand()) {
    mutable_symbol_operand()->mutable_operand()->__Init__(proto.symbol_operand());
  } else if (proto.has_init_symbol_operand()) {
    mutable_init_symbol_operand()->mutable_operand()->__Init__(proto.init_symbol_operand());
  } else if (proto.has_separator()) {
    mutable_separator();
  } else if (proto.has_double_operand()) {
    set_double_operand(proto.double_operand());
  } else if (proto.has_int64_operand()) {
    set_int64_operand(proto.int64_operand());
  } else if (proto.has_uint64_operand()) {
    set_uint64_operand(proto.uint64_operand());
  } else if (proto.has_bool_operand()) {
    set_bool_operand(proto.bool_operand());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace vm
}  // namespace oneflow
