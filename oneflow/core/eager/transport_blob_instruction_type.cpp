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
#include "oneflow/core/eager/transport_blob_instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace eager {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(SendBlobInstruction);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, src_blob);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, header_token_sep);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(uint64_t, header_token);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, body_token_sep);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(uint64_t, body_token);
FLAT_MSG_VIEW_END(SendBlobInstruction);

FLAT_MSG_VIEW_BEGIN(ReceiveBlobInstruction);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::Mut2Operand, dst_blob);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, header_token_sep);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(uint64_t, header_token);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, body_token_sep);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(uint64_t, body_token);
FLAT_MSG_VIEW_END(ReceiveBlobInstruction);
// clang-format on

}  // namespace

void SendBlobInstructionType::Compute(vm::Instruction* instruction) const {
  CHECK_OK(Send(instruction));
}

void ReceiveBlobInstructionType::Compute(vm::Instruction* instruction) const {
  CHECK_OK(Receive(instruction));
}

// Sends data to dst_machine
Maybe<void> SendBlobInstructionType::Send(vm::Instruction* instruction) const {
  FlatMsgView<SendBlobInstruction> args(instruction->instr_msg().operand());
  CHECK_EQ_OR_RETURN(args->src_blob_size(), args->header_token_size());
  CHECK_EQ_OR_RETURN(args->body_token_size(), args->header_token_size());
  RefCntType* ref_cnt = nullptr;
  {
    char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    ref_cnt = reinterpret_cast<RefCntType*>(data_ptr);
    *ref_cnt = 2 * args->src_blob_size();
  }
  // `ref_cnt` is safe to be captured before `Callback` finished.
  auto Callback = [ref_cnt] { CHECK_GE(--*ref_cnt, 0); };
  // Streams each take responsibility for destination machines.
  // See TransportStreamType::MakeTransportStreamType for more details.
  int64_t dst_machine_id = instruction->stream().device_id();
  int64_t this_machine_id = instruction->stream().machine_id();
  FOR_RANGE(int64_t, i, 0, args->src_blob_size()) {
    const char* header_mem_ptr = nullptr;
    std::size_t header_size = 0;
    const char* body_mem_ptr = nullptr;
    std::size_t body_size = 0;
    {
      const auto* src_blob_operand = instruction->operand_type(args->src_blob(i));
      CHECK_NOTNULL_OR_RETURN(src_blob_operand)
          << Error::RwMutexedObjectNotFoundError()
          << "src_blob: " << args->src_blob(i).logical_object_id() << ". "
          << "Did you forget broadcast the src_blob?";
      const auto& blob = JUST(src_blob_operand->template Get<BlobObject>())->blob();
      header_mem_ptr = blob.header_ptr();
      header_size = blob.blob_desc().ByteSizeOfBlobHeader();
      body_mem_ptr = blob.dptr<char>();
      // get actual byte size of blob body
      body_size = blob.ByteSizeOfBlobBody();
    }
    JUST(Send(this_machine_id, dst_machine_id, args->header_token(i), header_mem_ptr, header_size,
              Callback));
    JUST(Send(this_machine_id, dst_machine_id, args->body_token(i), body_mem_ptr, body_size,
              Callback));
  }
  return Maybe<void>::Ok();
}

// Receives data from src_machine
Maybe<void> ReceiveBlobInstructionType::Receive(vm::Instruction* instruction) const {
  FlatMsgView<ReceiveBlobInstruction> args(instruction->instr_msg().operand());
  CHECK_EQ_OR_RETURN(args->dst_blob_size(), args->header_token_size());
  CHECK_EQ_OR_RETURN(args->body_token_size(), args->header_token_size());
  RefCntType* ref_cnt = nullptr;
  {
    char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    ref_cnt = reinterpret_cast<RefCntType*>(data_ptr);
    *ref_cnt = 2 * args->dst_blob_size();
  }
  // `ref_cnt` is safe to be captured before `Callback` finished.
  auto Callback = [ref_cnt] { CHECK_GE(--*ref_cnt, 0); };
  // Streams each take responsibility for source machines.
  // See TransportStreamType::MakeTransportStreamType for more details.
  int64_t src_machine_id = instruction->stream().device_id();
  int64_t this_machine_id = instruction->stream().machine_id();
  FOR_RANGE(int64_t, i, 0, args->dst_blob_size()) {
    char* header_mem_ptr = nullptr;
    std::size_t header_size = 0;
    char* body_mem_ptr = nullptr;
    std::size_t body_size = 0;
    {
      auto* dst_blob_operand = instruction->mut_operand_type(args->dst_blob(i));
      CHECK_NOTNULL_OR_RETURN(dst_blob_operand)
          << Error::RwMutexedObjectNotFoundError()
          << "dst_blob: " << args->dst_blob(i).logical_object_id() << ". "
          << "Did you forget broadcast the dst_blob?";
      auto* blob = JUST(dst_blob_operand->template Mut<BlobObject>())->mut_blob();
      header_mem_ptr = blob->mut_header_ptr();
      header_size = blob->blob_desc().ByteSizeOfBlobHeader();
      body_mem_ptr = blob->mut_dptr<char>();
      // get capacity byte size of blob body
      body_size = blob->blob_desc().ByteSizeOfBlobBody();
    }
    JUST(Receive(src_machine_id, this_machine_id, args->header_token(i), header_mem_ptr,
                 header_size, Callback));
    JUST(Receive(src_machine_id, this_machine_id, args->body_token(i), body_mem_ptr, body_size,
                 Callback));
  }
  return Maybe<void>::Ok();
}

Maybe<void> SendBlobInstructionType::Send(int64_t this_machine_id, int64_t dst_machine_id,
                                          uint64_t body_token, const char* mem_ptr,
                                          std::size_t size,
                                          const std::function<void()>& Callback) const {
  TODO();
  return Maybe<void>::Ok();
}

Maybe<void> ReceiveBlobInstructionType::Receive(int64_t src_machine_id, int64_t this_machine_id,
                                                uint64_t body_token, char* mem_ptr,
                                                std::size_t size,
                                                const std::function<void()>& Callback) const {
  TODO();
  return Maybe<void>::Ok();
}

COMMAND(vm::RegisterInstructionType<SendBlobInstructionType>("SendBlob"));
COMMAND(vm::RegisterInstructionType<ReceiveBlobInstructionType>("ReceiveBlob"));

}  // namespace eager
}  // namespace oneflow
