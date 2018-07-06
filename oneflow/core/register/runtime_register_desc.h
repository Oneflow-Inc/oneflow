#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class Regst;

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
  const BlobDesc* GetBlobDescFromLbi(const LogicalBlobId& lbi) const;
  const std::vector<LogicalBlobId>& sorted_lbis() const { return sorted_lbis_; };
  const BlobDesc* packed_blob_desc() const { return &packed_blob_desc_; }

  void AccumulateActuallyMemCaseSize(const BlobDesc* blob_desc);
  const HashMap<MemoryCase, size_t>& GetSize4AllActuallyMemCase() const {
    return mem_case2mem_size_;
  }
  HashMap<MemoryCase, char*> GetMemPtrOfMemCase4Regst(int64_t regst_index) const;
  void PickMemory(const MemoryCase& mem_case, char* mem_ptr);
  void PickOrOccupyMemoryFromMemBlock(HashMap<MemoryCase, char*>& mem_case2mem_ptr, bool occupy);

 private:
  int64_t regst_desc_id_;
  int64_t producer_actor_id_;
  std::vector<int64_t> consumers_actor_id_;
  int64_t register_num_;
  MemoryCase mem_case_;
  int32_t mem_shared_id_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2blob_desc_;
  std::vector<LogicalBlobId> sorted_lbis_;
  BlobDesc packed_blob_desc_;
  HashMap<MemoryCase, size_t> mem_case2mem_size_;
  HashMap<MemoryCase, char*> mem_case2mem_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
