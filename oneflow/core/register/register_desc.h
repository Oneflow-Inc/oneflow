#ifndef ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_

#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

const int32_t kMaxRegisterNum = std::numeric_limits<int32_t>::max();

class TaskNode;

class RegstDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDesc);
  RegstDesc();
  ~RegstDesc() = default;

  // regst_desc_id
  int64_t regst_desc_id() const { return regst_desc_id_; }

  // producer_, consumers_
  const TaskNode* producer() const { return producer_; }
  void set_producer(const TaskNode* val) { producer_ = val; }
  const HashSet<const TaskNode*>& consumers() const { return consumers_; }
  void AddConsumer(const TaskNode*);

  // min_register_num_, max_register_num_
  int32_t min_register_num() const { return min_register_num_; }
  void set_min_register_num(int32_t val);
  int32_t max_register_num() const { return max_register_num_; }
  void set_max_register_num(int32_t val);

  // lbn2blob_desc_
  bool IsLocked() const { return is_locked_; }
  void Lock();
  void CopyBlobDescFrom(const RegstDesc*);
  void CopyBlobDescWithoutAddLbn(const RegstDesc*);
  BlobDesc* AddLbn(const std::string& lbn);
  const BlobDesc* GetBlobDesc(const std::string& lbn) const;
  BlobDesc* MutBlobDesc(const std::string& lbn);
  void ForEachLbn(std::function<void(const std::string&)> func) const;
  size_t NumOfLbn() const { return lbn2blob_desc_.size(); }

  // util
  int32_t MaxSeqSize() const { return packed_blob_desc_->max_seq_size(); }
  void InferMemCase();
  void EraseZeroSizeBlob();
  void ToProto(RegstDescProto*) const;

 private:
  int64_t regst_desc_id_;
  const TaskNode* producer_;
  HashSet<const TaskNode*> consumers_;
  int32_t min_register_num_;
  int32_t max_register_num_;

  HashMap<std::string, std::unique_ptr<BlobDesc>> lbn2blob_desc_;
  std::unique_ptr<BlobDesc> packed_blob_desc_;
  bool is_locked_;

  MemoryCase mem_case_;
};

inline void UpdtMinRegisterNumByMaxSequenceSize(RegstDesc* regst_desc) {
  if (regst_desc->min_register_num() < register_desc->MaxSeqSize()) {
    regst_desc->set_min_register_num(regst_desc->MaxSeqSize());
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
