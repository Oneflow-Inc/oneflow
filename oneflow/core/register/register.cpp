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
#include "oneflow/core/register/register.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

const std::vector<int64_t>& Regst::consumers_actor_id() const {
  return regst_desc_->consumers_actor_id();
}

Regst::Regst()
    : regst_desc_(nullptr),
      main_mem_ptr_(nullptr),
      separated_header_mem_ptr_(nullptr),
      comm_net_token_(nullptr) {}

Regst::~Regst() {
  if (comm_net_token_ != nullptr) { Singleton<CommNet>::Get()->UnRegisterMemory(comm_net_token_); }
}

Blob* Regst::GetBlobByOrdinal(int64_t ordinal) { return sorted_blob_vec_.at(ordinal).get(); }

Blob* Regst::GetBlobByLbi(const LogicalBlobId& lbi) {
  const int64_t ordinal = regst_desc_->GetOrdinalForLbi(lbi);
  if (ordinal >= 0) {
    return sorted_blob_vec_.at(ordinal).get();
  } else {
    return nullptr;
  }
}

void Regst::set_regst_desc(const RtRegstDesc* regst_desc) {
  CHECK(regst_desc_ == nullptr);
  regst_desc_ = regst_desc;
  sorted_blob_vec_.resize(regst_desc->lbi_num());
}

void Regst::SetBlobByOrdinal(int64_t ordinal, std::unique_ptr<Blob>&& blob) {
  CHECK(!sorted_blob_vec_.at(ordinal));
  sorted_blob_vec_.at(ordinal).swap(blob);
}

Blob* Regst::GetMutSoleBlob() {
  CHECK_EQ(GetBlobSize(), 1);
  return sorted_blob_vec_.front().get();
}

const Blob* Regst::GetSoleBlob() const {
  CHECK_EQ(GetBlobSize(), 1);
  return sorted_blob_vec_.front().get();
}

void* Regst::comm_net_token() {
  void* token = comm_net_token_.load(std::memory_order_relaxed);
  if (token != nullptr) { return token; }
  {
    std::lock_guard<std::mutex> lock(comm_net_token_mutex_);
    token = comm_net_token_;
    if (token != nullptr) { return token; }
    CHECK(main_mem_ptr() != nullptr);
    CHECK(separated_header_mem_ptr() == nullptr);
    token = Singleton<CommNet>::Get()->RegisterMemory(main_mem_ptr(),
                                                      this->regst_desc()->MainByteSize4OneRegst());
    comm_net_token_ = token;
    return token;
  }
}

}  // namespace oneflow
