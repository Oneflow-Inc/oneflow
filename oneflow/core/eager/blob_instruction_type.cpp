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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/access_blob_arg_cb_phy_instr_operand.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace vm {

void AccessBlobByCallbackInstructionType::Compute(vm::Instruction* instruction) const {
  const auto& phy_instr_operand = instruction->phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::AccessBlobArgCbPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  StreamPolicy* stream_policy = instruction->mut_stream_policy();
  OfBlob ofblob(stream_policy->stream(), ptr->eager_blob_object()->blob());
  ptr->callback()(reinterpret_cast<uint64_t>(&ofblob));
}

}  // namespace vm
}  // namespace oneflow
