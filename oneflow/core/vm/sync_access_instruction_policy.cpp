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
#include "oneflow/core/vm/sync_access_instruction_policy.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace vm {

SyncAccessInstructionPolicy::SyncAccessInstructionPolicy()
    : host_mem_case_(memory::MakeHostMemCase()),
      btb_(),
      mem_ptr_(nullptr),
      bytes_(0),
      eager_blob_object_(nullptr) {
  ResetBase(nullptr, 0, nullptr);
}

void SyncAccessInstructionPolicy::ResetBase(char* mem_ptr, size_t bytes,
                                            EagerBlobObject* eager_blob_object) {
  btb_.Reset();
  mem_ptr_ = mem_ptr;
  bytes_ = bytes;
  eager_blob_object_ = eager_blob_object;
}

namespace {

void FastCopy(char* dst, const char* src, size_t bytes) {
  switch (bytes) {
    case 1: {
      *dst = *src;
      return;
    }
    case 2: {
      *reinterpret_cast<int16_t*>(dst) = *reinterpret_cast<const int16_t*>(src);
      return;
    }
    case 4: {
      *reinterpret_cast<int32_t*>(dst) = *reinterpret_cast<const int32_t*>(src);
      return;
    }
    case 8: {
      *reinterpret_cast<int64_t*>(dst) = *reinterpret_cast<const int64_t*>(src);
      return;
    }
    case 16: {
      using Bit128 = std::pair<int64_t, int64_t>;
      *reinterpret_cast<Bit128*>(dst) = *reinterpret_cast<const Bit128*>(src);
      return;
    }
    default: UNIMPLEMENTED() << "FastCopy on bytes " << bytes << " not supported.";
  }
}

}  // namespace

void SyncReadInstructionPolicy::Compute(Instruction* instruction) {
  StreamPolicy* stream_policy = instruction->mut_stream_policy();
  char* pinned_buffer = instruction->mut_stream()->CheckSizeAndGetTmpSmallPinnedMemPtr(bytes_);
  mut_btb()->mut_notifier()->Notify();
  SyncAutoMemcpy(stream_policy->stream(), pinned_buffer, eager_blob_object_->mut_dptr(), bytes_,
                 host_mem_case_, eager_blob_object_->mem_case());
  FastCopy(mem_ptr_, pinned_buffer, bytes_);
  mut_btb()->mut_spin_counter()->Decrease();
}

}  // namespace vm
}  // namespace oneflow
