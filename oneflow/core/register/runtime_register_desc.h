#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/blob_desc.h"
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
  const std::vector<int64_t>& consumers_actor_id() const {
    return consumers_actor_id_;
  }
  int64_t register_num() const { return register_num_; }
  const MemoryCase& mem_case() const { return mem_case_; }

  const BlobDesc* GetBlobDescFromLbn(const std::string& lbn) const;
  const BlobDesc* packed_blob_desc() const { return &packed_blob_desc_; }

 private:
  int64_t regst_desc_id_;
  int64_t producer_actor_id_;
  std::vector<int64_t> consumers_actor_id_;
  int64_t register_num_;
  MemoryCase mem_case_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> lbn2blob_desc_;
  BlobDesc packed_blob_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
