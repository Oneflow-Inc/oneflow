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
#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_

#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/operand_def.h"

namespace oneflow {
namespace eager {

// clang-format off
FLAT_MSG_VIEW_BEGIN(NewOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::SymbolOperand, job_desc);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, op_conf);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, op);
FLAT_MSG_VIEW_END(NewOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(DeleteOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, op);
FLAT_MSG_VIEW_END(DeleteOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(CallOpKernelInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::MutOperand, opkernel);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::SymbolOperand, op_node_signature);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_const_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, const_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::ConstOperand, const_input_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_mut_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, mut_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, mut_input_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, output_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_mut2_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, mut2_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::Mut2Operand, mut2_output_blob);
FLAT_MSG_VIEW_END(CallOpKernelInstrOperand);

FLAT_MSG_VIEW_BEGIN(StatelessCallOpKernelInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::SymbolOperand, job_desc);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::SymbolOperand, op_conf);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::SymbolOperand, op_node_signature);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::MutOperand, shared_opkernel);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_const_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, const_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::ConstOperand, const_input_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_mut_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, mut_ibn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, mut_input_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::MutOperand, output_blob);

  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::OperandSeparator, begin_mut2_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::SymbolOperand, mut2_obn);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::Mut2Operand, mut2_output_blob);
FLAT_MSG_VIEW_END(StatelessCallOpKernelInstrOperand);

FLAT_MSG_VIEW_BEGIN(FetchBlobInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::ConstOperand, blob);
  FLAT_MSG_VIEW_DEFINE_OPERAND(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(FetchBlobInstrOperand);

FLAT_MSG_VIEW_BEGIN(FeedBlobInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::Mut2Operand, blob);
  FLAT_MSG_VIEW_DEFINE_OPERAND(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(FeedBlobInstrOperand);

FLAT_MSG_VIEW_BEGIN(RemoveForeignCallbackInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND(vm::MutOperand, object_id);
  FLAT_MSG_VIEW_DEFINE_OPERAND(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(RemoveForeignCallbackInstrOperand);
// clang-format on

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
