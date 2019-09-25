#ifndef ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_

#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

const int32_t kMaxRegisterNum = std::numeric_limits<int32_t>::max();

void InitCtrlRegstDesc(int64_t producer_task_id, RegstDescProto* ctrl_regst_proto);

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
  void DeleteConsumer(const TaskNode*);

  // min_register_num_, max_register_num_
  int32_t min_register_num() const { return min_register_num_; }
  void UpdtMinRegstNumIfNeed(int32_t val);
  int32_t max_register_num() const { return max_register_num_; }
  void UpdtMaxRegstNumIfNeed(int32_t val);

  // lbi2blob_desc_
  bool IsLocked() const { return is_locked_; }
  void Lock();
  void CopyBlobDescFrom(const RegstDesc*);
  void CopyBlobDescWithoutAddLbi(const RegstDesc*);
  BlobDesc* AddLbi(const LogicalBlobId&);
  const BlobDesc* GetBlobDesc(const LogicalBlobId& lbi) const;
  bool HasLbi(const LogicalBlobId& lbi) const;
  BlobDesc* MutBlobDesc(const LogicalBlobId& lbi);
  const BlobDesc* SoleBlobDesc() const;
  BlobDesc* MutSoleBlobDesc();
  void ForEachLbi(std::function<void(const LogicalBlobId&)> func) const;
  size_t NumOfLbi() const { return lbi2blob_desc_.size(); }

  // mem
  const MemoryCase& mem_case() const { return mem_case_; }
  MemoryCase* mut_mem_case() { return &mem_case_; }
  bool enable_reuse_mem() { return enable_reuse_mem_; }
  void set_enable_reuse_mem(bool enable_reuse_mem) { enable_reuse_mem_ = enable_reuse_mem; }
  int64_t mem_block_offset() const;
  void set_mem_block_offset(int64_t val) { mem_block_offset_ = val; }
  void set_hint_inplace_consumed_regst_desc_id(int64_t val) {
    hint_inplace_consumed_regst_desc_id_ = val;
  }
  int32_t mem_block_id() const { return mem_block_id_; }
  void set_mem_block_id(int32_t val) { mem_block_id_ = val; }
  bool HasSetMemBlockId() { return mem_block_id_ != -1; }
  void CopyMemBlockInfoFrom(const RegstDesc*);

  const std::shared_ptr<Shape>& data_regst_time_shape() const {
    CHECK(regst_desc_type_.has_data_regst_desc());
    CHECK(data_regst_time_shape_);
    return data_regst_time_shape_;
  }
  std::shared_ptr<Shape>* mut_data_regst_time_shape() {
    CHECK(regst_desc_type_.has_data_regst_desc());
    return &data_regst_time_shape_;
  }
  RegstDescTypeProto* mut_regst_desc_type() { return &regst_desc_type_; }
  const RegstDescTypeProto& regst_desc_type() const { return regst_desc_type_; }
  bool HasSameMemSize(const RegstDesc*);

  // util
  int32_t MaxColNum() const { return packed_blob_desc_->max_col_num(); }
  void EraseZeroSizeBlob();
  void ToProto(RegstDescProto*) const;
  bool HasSameBlobDescs(const RegstDesc*);
  int64_t ByteOffsetInPackedBlobDescBody(const LogicalBlobId& lbi) const;

 private:
  int64_t regst_desc_id_;
  const TaskNode* producer_;
  HashSet<const TaskNode*> consumers_;
  int32_t min_register_num_;
  int32_t max_register_num_;

  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2blob_desc_;
  std::unique_ptr<BlobDesc> packed_blob_desc_;
  bool is_locked_;

  MemoryCase mem_case_;
  RegstDescTypeProto regst_desc_type_;
  bool enable_reuse_mem_;
  int32_t mem_block_id_;
  int64_t mem_block_offset_;
  int32_t hint_inplace_consumed_regst_desc_id_;

  std::shared_ptr<Shape> data_regst_time_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
