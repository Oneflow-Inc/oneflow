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
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/eager/lazy_ref_blob_object.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/vm/access_blob_arg_cb_phy_instr_operand.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/tensor_view_operand.h"

namespace oneflow {
namespace vm {

void TensorViewInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr = dynamic_cast<const vm::TensorViewOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  OfBlob input_ofblob(device_ctx->stream(), ptr->eager_blob_object()->mut_blob());
  OfBlob view_ofblob(device_ctx->stream(), ptr->view_eager_blob_object()->mut_blob());

  void* input_ptr = input_ofblob.mut_blob()->mut_raw_dptr();
  view_ofblob.mut_blob()->reset_dptr(static_cast<char*>(input_ptr));
}

void AccessBlobByCallbackInstructionType::Compute(vm::Instruction* instruction) const {
  const vm::InstructionMsg& instr_msg = instruction->instr_msg();
  const auto& phy_instr_operand = instr_msg.phy_instr_operand();
  CHECK(static_cast<bool>(phy_instr_operand));
  const auto* ptr =
      dynamic_cast<const vm::AccessBlobArgCbPhyInstrOperand*>(phy_instr_operand.get());
  CHECK_NOTNULL(ptr);
  DeviceCtx* device_ctx = instruction->stream().device_ctx().get();
  OfBlob ofblob(device_ctx->stream(), ptr->eager_blob_object()->mut_blob());
  ptr->callback()(reinterpret_cast<uint64_t>(&ofblob));
}

}  // namespace vm
}  // namespace oneflow
