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
  void set_regst_desc_id(int64_t val) { regst_desc_id_ = val; }

  //
  int32_t min_register_num() const { return min_register_num_; }
  void set_min_register_num(int32_t val) { min_register_num_ = val; }
  int32_t max_register_num() const { return max_register_num_; }
  void set_max_register_num(int32_t val) { max_register_num_ = val; }

  // Producer
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }
  const HashSet<const TaskNode*>& consumers() const { return consumers_; }
  void AddConsumer(const TaskNode*);

  // Lbn and BlobDesc
  void CopyLbnFrom(const RegstDesc*);
  void CopyBlobDescFrom(const RegstDesc*);
  void EnrollLbn(const std::string& lbn);
  const BlobDesc& GetBlobDesc(const std::string& lbn) const;
  BlobDesc* GetMutBlobDesc(const std::string& lbn);
  void ForEachLbn(std::function<void(const std::string&)> func) const;
  size_t NumOfLbn() const { return lbn2blob_desc_.size(); }

  //
  void EraseZeroSizeBlob();
  void EraseBlobDesc(const std::string& lbn);
  std::string DebugStr() const;
  void ToProto(RegstDescProto*) const;
  MemoryCase InferMemCase() const;
  BlobDesc CompPackedBlobDesc() const;

 private:
  int64_t CompElemCntOfAllBlob() const;
  int64_t regst_desc_id_;
  const TaskNode* producer_;
  HashSet<const TaskNode*> consumers_;
  int32_t min_register_num_;
  int32_t max_register_num_;

  HashMap<std::string, std::unique_ptr<BlobDesc>> lbn2blob_desc_;
  int64_t register_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
