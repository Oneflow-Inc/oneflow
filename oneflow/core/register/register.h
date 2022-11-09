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

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst();

  // Getters
  int64_t regst_desc_id() const {
    CHECK(regst_desc_ != nullptr);
    return regst_desc_->regst_desc_id();
  }

  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;
  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  Blob* GetBlobByOrdinal(int64_t ordinal);
  Blob* GetBlobByLbi(const LogicalBlobId& lbi);
  const Blob* GetSoleBlob() const;
  Blob* GetMutSoleBlob();
  int64_t GetBlobSize() const { return static_cast<int64_t>(sorted_blob_vec_.size()); }

  void* main_mem_ptr() const { return main_mem_ptr_; }
  void set_main_mem_ptr(void* ptr) { main_mem_ptr_ = ptr; }
  void* separated_header_mem_ptr() const { return separated_header_mem_ptr_; }
  void set_separated_header_mem_ptr(void* ptr) { separated_header_mem_ptr_ = ptr; }
  void* comm_net_token();

 private:
  friend class RegstMgr;
  Regst();

  void set_regst_desc(const RtRegstDesc* regst_desc);
  void SetBlobByOrdinal(int64_t ordinal, std::unique_ptr<Blob>&& blob);

  const RtRegstDesc* regst_desc_;
  std::vector<std::unique_ptr<Blob>> sorted_blob_vec_;
  void* main_mem_ptr_;
  void* separated_header_mem_ptr_;

  std::atomic<void*> comm_net_token_;
  std::mutex comm_net_token_mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
