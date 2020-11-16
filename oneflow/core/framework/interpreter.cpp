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
#include "oneflow/core/framework/interpreter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/eager/eager_oneflow.h"

namespace oneflow {

LogicalInterpreter::LogicalInterpreter()
    : Interpreter(std::make_shared<vm::LogicalIdGenerator>()) {}

Maybe<void> LogicalInterpreter::Run(const std::function<Maybe<void>(InstructionsBuilder*)>& Build) {
  InstructionsBuilder instructions_builder(mut_id_generator());
  JUST(Build(&instructions_builder));
  if (instructions_builder.instruction_list().instruction().empty()) {
    CHECK(instructions_builder.eager_symbol_list().eager_symbol().empty());
    return Maybe<void>::Ok();
  }
  return Global<eager::EagerOneflow>::Get()->RunLogicalInstruction(
      instructions_builder.instruction_list(), instructions_builder.eager_symbol_list());
}

PhysicalInterpreter::PhysicalInterpreter()
    : Interpreter(std::make_shared<vm::PhysicalIdGenerator>()) {}

Maybe<void> PhysicalInterpreter::Run(
    const std::function<Maybe<void>(InstructionsBuilder*)>& Build) {
  InstructionsBuilder instructions_builder(mut_id_generator());
  JUST(Build(&instructions_builder));
  if (instructions_builder.instruction_list().instruction().empty()) {
    CHECK(instructions_builder.eager_symbol_list().eager_symbol().empty());
    return Maybe<void>::Ok();
  }
  return Global<eager::EagerOneflow>::Get()->RunPhysicalInstruction(
      instructions_builder.instruction_list(), instructions_builder.eager_symbol_list());
}

}  // namespace oneflow
