#ifndef ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

// Regst  : Register
// Contig : Contiguous

class TaskNode;

class RegstDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDesc);
  RegstDesc();
  ~RegstDesc() = default;

  // regst_desc_id
  int64_t regst_desc_id() const { return regst_desc_id_; }
  void set_regst_desc_id(int64_t val) { regst_desc_id_ = val; }
  // Producer
  const TaskNode* GetProducer() const { return producer_; }
  void SetProducer(const TaskNode* task_node) { producer_ = task_node; }
  const HashSet<const TaskNode*>& subscribers() const { return subscribers_; }
  void AddSubscriber(const TaskNode*);

  // Lbn and Shape
  void CopyLbnFrom(const RegstDesc*);
  void CopyShapeFrom(const RegstDesc*);
  void EnrollLbn(const std::string& lbn);
  const Shape& GetShape(const std::string& lbn) const;
  Shape* GetMutShapePtr(const std::string& lbn);
  void ForEachLbn(std::function<void(const std::string&)> func) const;
  size_t NumOfLbn() const { return lbn2shape_.size(); }

  //
  void EraseZeroSizeBlob();
  int64_t CompElemCntOfAllBlob() const;
  std::string DebugStr() const;
  void ToProto(RegstDescProto*) const;
  MemoryCase InferMemCase() const;
  
 private:
  int64_t regst_desc_id_;
  const TaskNode* producer_;
  HashSet<const TaskNode*> subscribers_;
  
  HashMap<std::string, std::unique_ptr<Shape>> lbn2shape_;
  int64_t register_num_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REGISTER_DESC_H_
