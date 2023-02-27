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
#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

enum class RegstAllocationType {
  kInvalid = 0,
  kStatic = 1,
  kStreamOrdered = 2,
};

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst();

  // Getters
  int64_t regst_desc_id() const {
    CHECK(regst_desc_ != nullptr);
    return regst_desc_->regst_desc_id();
  }

  void Init(void* header_mem_ptr);
  void ResetBodyMemPtr(void* body_mem_ptr);
  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;
  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  Blob* GetBlobByOrdinal(int64_t ordinal);
  Blob* GetBlobByLbi(const LogicalBlobId& lbi);
  const Blob* GetSoleBlob() const;
  Blob* GetMutSoleBlob();
  int64_t GetBlobSize() const { return static_cast<int64_t>(sorted_blob_vec_.size()); }

  void* comm_net_token();

  void* header_mem_ptr() const { return header_mem_ptr_; }

  void* body_mem_ptr() const { return body_mem_ptr_; }

  RegstAllocationType allocation_type() const { return allocation_type_; }

 private:
  friend class RegstMgr;
  Regst(const RtRegstDesc* regst_desc, RegstAllocationType allocation_type);

  void SetBlobByOrdinal(int64_t ordinal, std::unique_ptr<Blob>&& blob);

  const RtRegstDesc* regst_desc_;
  std::vector<std::unique_ptr<Blob>> sorted_blob_vec_;

  void* header_mem_ptr_;
  void* body_mem_ptr_;

  std::atomic<void*> comm_net_token_;
  std::mutex comm_net_token_mutex_;
  RegstAllocationType allocation_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
