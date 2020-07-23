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
#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class RtRegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtRegstDesc);
  RtRegstDesc() = delete;
  ~RtRegstDesc() = default;

  RtRegstDesc(const RegstDescProto& regst_desc_proto);

  int64_t regst_desc_id() const { return regst_desc_id_; }
  int64_t producer_actor_id() const { return producer_actor_id_; }
  const std::vector<int64_t>& consumers_actor_id() const { return consumers_actor_id_; }
  int64_t register_num() const { return register_num_; }
  const MemoryCase& mem_case() const { return mem_case_; }
  const RegstDescTypeProto& regst_desc_type() const { return regst_desc_type_; }

  const RtBlobDesc* GetRtBlobDescFromLbi(const LogicalBlobId& lbi) const;
  const RtBlobDesc* packed_blob_desc() const { return packed_blob_desc_.get(); }
  size_t TotalByteSize4AllRegst() const;
  size_t TotalMainByteSize4AllRegst() const;
  size_t TotalSeparatedHeaderByteSize4AllRegst() const;
  size_t SeparatedHeaderByteSize4OneRegst() const;
  size_t MainByteSize4OneRegst() const;
  const Shape& data_regst_time_shape() const;
  bool is_body_disabled() const { return packed_blob_desc_->is_body_disabled(); }

  void ForEachBlobDescOffsetInOnRegst(
      const std::vector<LbiBlobDescPair>& lbis,
      const std::function<void(const LbiBlobDescPair&, int64_t body_offset, int64_t header_offset)>&
          Handler) const;

 private:
  int64_t regst_desc_id_;
  int64_t producer_actor_id_;
  std::vector<int64_t> consumers_actor_id_;
  int64_t register_num_;
  RegstDescTypeProto regst_desc_type_;
  MemoryCase mem_case_;
  HashMap<LogicalBlobId, std::unique_ptr<RtBlobDesc>> lbi2blob_desc_;
  std::unique_ptr<RtBlobDesc> packed_blob_desc_;
  std::unique_ptr<Shape> data_regst_time_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
